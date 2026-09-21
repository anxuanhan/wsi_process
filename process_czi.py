import argparse
import openslide
import cv2
from PIL import Image
import numpy as np
import os

def read_czi_rgb(czi_path, level=6):
    try:
        from pylibCZIrw import czi
    except ImportError as exc:
        raise ImportError(
            "Reading .czi files requires pylibCZIrw in the environment running this script."
        ) from exc

    zoom = 1 / (2 ** level)
    with czi.open_czi(czi_path) as czidoc:
        img = czidoc.read(zoom=zoom)

    img = np.asarray(img)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] > 3:
        img = img[..., :3]

    # pylibCZIrw returns BGR here, matching the notebook preview code.
    return img[..., ::-1].copy(), zoom

def keep_border_connected_regions(candidate_mask, min_area=0):
    candidate_mask = candidate_mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, 8)
    if n <= 1:
        return np.zeros_like(candidate_mask, dtype=bool)

    border_labels = np.unique(
        np.concatenate([
            labels[0, :],
            labels[-1, :],
            labels[:, 0],
            labels[:, -1],
        ])
    )
    border_labels = border_labels[border_labels != 0]

    keep = np.zeros_like(candidate_mask, dtype=bool)
    for label in border_labels:
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep |= labels == label
    return keep

def build_gray_background_mask(rgb_array, gray_min_value=130, gray_max_saturation=55):
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    mx = rgb_array.max(axis=2)
    mn = rgb_array.min(axis=2)

    near_neutral = (mx - mn) < 35
    light_enough = hsv[:, :, 2] > gray_min_value
    low_saturation_gray = (hsv[:, :, 1] < gray_max_saturation) & light_enough & near_neutral
    neutral_light_gray = (mn > 180) & ((mx - mn) < 25)
    gray_candidate = neutral_light_gray | low_saturation_gray

    return keep_border_connected_regions(gray_candidate)

def clean_czi_preview(rgb_array, black_threshold=10, min_black_area=100, gray_min_value=130, gray_max_saturation=55):
    img = rgb_array.copy()
    dark = np.all(img < black_threshold, axis=2)
    black_bg = keep_border_connected_regions(dark, min_black_area)
    img[black_bg] = 255

    gray_bg = build_gray_background_mask(img, gray_min_value, gray_max_saturation)
    img[gray_bg] = 255

    return img

def clean_light_gray_background(rgb_array, gray_min_value=130, gray_max_saturation=55):
    img = rgb_array.copy()
    gray_bg = build_gray_background_mask(img, gray_min_value, gray_max_saturation)
    img[gray_bg] = 255
    return img

def brighten_image(rgb_array, brightness=20):
    if brightness == 0:
        return rgb_array.copy()

    lab = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0].astype(np.int16)
    lab[:, :, 0] = np.clip(l_channel + brightness, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def build_large_component_mask(candidate_mask, min_pixels):
    candidate_mask = candidate_mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_mask, 8)
    keep = np.zeros_like(candidate_mask, dtype=bool)

    for label in range(1, n):
        if stats[label, cv2.CC_STAT_AREA] > min_pixels:
            keep |= labels == label
    return keep

def build_black_pen_mark_mask(rgb_array, min_pixels=100, black_threshold=110):
    black_mask = np.all(rgb_array < black_threshold, axis=2)
    return build_large_component_mask(black_mask, min_pixels)

def remove_pen_marks_fast(rgb_array, min_pixels=100, black_threshold=110, kernel_size=20, dilate_iterations=1):
    if min_pixels <= 0:
        return rgb_array.copy()

    pen_mask = build_black_pen_mark_mask(rgb_array, min_pixels, black_threshold).astype(np.uint8) * 255
    if kernel_size > 0 and dilate_iterations > 0 and np.any(pen_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        pen_mask = cv2.dilate(pen_mask, kernel, iterations=dilate_iterations)

    img = rgb_array.copy()
    img[pen_mask > 0] = 255
    return img

def find_tissue_and_hole_contours(rgb_array, min_area, min_hole_area, downsample_factor, min_fragment_ratio=0.03):
    gray = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    binary_opened = cv2.morphologyEx(binary_closed, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, hierarchy = cv2.findContours(binary_opened, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    adjusted_min_area = min_area / (downsample_factor ** 2)
    adjusted_min_hole_area = min_hole_area / (downsample_factor ** 2)
    outer_contours = []
    hole_contours = []

    if hierarchy is not None:
        hierarchy = hierarchy[0]
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if hierarchy[i][3] == -1:
                if area > adjusted_min_area:
                    outer_contours.append(cnt)
            else:
                if area > adjusted_min_hole_area:
                    hole_contours.append(cnt)

        if len(hole_contours) > 10:
            hole_contours = sorted(hole_contours, key=cv2.contourArea, reverse=True)[:10]

        if len(outer_contours) > 0:
            areas = [cv2.contourArea(cnt) for cnt in outer_contours]
            max_area = max(areas)
            min_fragment_area = max_area * min_fragment_ratio
            outer_contours = [
                cnt for cnt in outer_contours
                if cv2.contourArea(cnt) >= min_fragment_area
            ]

    return binary_opened, outer_contours, hole_contours

def process_ndpi_files(input_folder, min_area, min_hole_area, out_dir="."):
    ndpi_files = [f for f in os.listdir(input_folder) if f.endswith(".ndpi")]

    for idx, ndpi_file in enumerate(ndpi_files, start=1):
        # 打开 ndpi 文件
        slide_path = os.path.join(input_folder, ndpi_file)
        slide = openslide.OpenSlide(slide_path)

        # 获取文件前缀名
        file_prefix = os.path.splitext(ndpi_file)[0]

        # 获取 Level 0 的尺寸
        level_0_dimensions = slide.level_dimensions[0]

        # 获取物理分辨率（每像素微米）
        mpp_x = slide.properties.get('openslide.mpp-x')
        mpp_y = slide.properties.get('openslide.mpp-y')
        if mpp_x and mpp_y:
            avg_mpp = (float(mpp_x) + float(mpp_y)) / 2
            magnification = "40x" if avg_mpp < 0.3 else "20x" if avg_mpp < 0.6 else "Unknown"
        else:
            magnification = "Unknown"

        # 创建保存图片的文件夹（由 out_dir 控制）
        origin_dir = os.path.join(out_dir, "origin_pic")
        circle_dir = os.path.join(out_dir, "circle_pic")
        mask_dir = os.path.join(out_dir, "mask_pic")
        os.makedirs(origin_dir, exist_ok=True)
        os.makedirs(circle_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        # 选择检测 level
        detection_level = min(3, slide.level_count - 1)
        level_dimensions = slide.level_dimensions[detection_level]

        # 从选定的 level 读取整个图像用于检测
        detection_img = slide.read_region((0, 0), detection_level, level_dimensions)
        detection_img = detection_img.convert('RGB')
        detection_array = np.array(detection_img)
        
        # 保存原始图像用于origin_pic
        detection_array_original = detection_array.copy()

        # 转换为灰度图
        gray = cv2.cvtColor(detection_array, cv2.COLOR_RGB2GRAY)

        # 使用 Otsu's 二值化方法
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # ===== 恢复原始参数：形态学操作 =====
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary_closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary_opened = cv2.morphologyEx(binary_closed, cv2.MORPH_OPEN, kernel, iterations=2)

        # 查找层级轮廓
        contours, hierarchy = cv2.findContours(binary_opened, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        # 根据检测 level 的分辨率调整面积阈值
        downsample_factor = slide.level_downsamples[detection_level]
        adjusted_min_area = min_area / (downsample_factor ** 2)
        adjusted_min_hole_area = min_hole_area / (downsample_factor ** 2)

        outer_contours = []
        hole_contours = []

        if hierarchy is not None:
            hierarchy = hierarchy[0]
            for i, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if hierarchy[i][3] == -1:  # 外轮廓
                    if area > adjusted_min_area:
                        outer_contours.append(cnt)
                else:  # 内部孔洞
                    if area > adjusted_min_hole_area:
                        hole_contours.append(cnt)

        # 在检测分辨率的图像上绘制轮廓
        result_detection = detection_array.copy()
        result_detection_bgr = cv2.cvtColor(result_detection, cv2.COLOR_RGB2BGR)

        # 绘制轮廓
        cv2.drawContours(result_detection_bgr, outer_contours, -1, (0, 255, 0), 3)
        cv2.drawContours(result_detection_bgr, hole_contours, -1, (255, 0, 0), 3)

        # 转换回 RGB
        result_detection_rgb = cv2.cvtColor(result_detection_bgr, cv2.COLOR_BGR2RGB)

        # 缩放到 800x800（保持长宽比）
        h, w = result_detection_rgb.shape[:2]
        scale = min(800 / w, 800 / h)
        new_w = int(w * scale)
        new_h = int(h * scale)

        result_resized = cv2.resize(result_detection_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建 800x800 的白色背景
        result_final = np.ones((800, 800, 3), dtype=np.uint8) * 240

        # 计算居中位置
        y_offset = (800 - new_h) // 2
        x_offset = (800 - new_w) // 2

        # 将缩放后的图像放到中心
        result_final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = result_resized

        # 保存带轮廓的图像
        result_pil = Image.fromarray(result_final)
        result_pil.save(os.path.join(circle_dir, f"{file_prefix}_tissue_contour_800x800.png"))

        # 同样的方法处理原始缩略图（使用未处理的原始图像）
        thumbnail_resized = cv2.resize(detection_array_original, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        thumbnail_final = np.ones((800, 800, 3), dtype=np.uint8) * 240
        thumbnail_final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = thumbnail_resized
        thumbnail_pil = Image.fromarray(thumbnail_final)
        thumbnail_pil.save(os.path.join(origin_dir, f"{file_prefix}_thumbnail_800x800.png"))

        # 保存 mask
        # 创建更新的mask：组织区域为白色，孔洞为黑色
        mask_with_holes = np.zeros_like(binary_opened)  # 初始化一个全黑的掩膜

        # 将外轮廓填充为白色（255）
        cv2.drawContours(mask_with_holes, outer_contours, -1, 255, -1)

        # 将孔洞区域填充为黑色（0）
        cv2.drawContours(mask_with_holes, hole_contours, -1, 0, -1)

        # 保存二值化图像到 mask_pic 文件夹
        binary_pil = Image.fromarray(mask_with_holes)
        binary_pil.save(os.path.join(mask_dir, f"{file_prefix}_binary_mask_800x800.png"))

        # 打印信息
        print(f"Processing file {idx}/{len(ndpi_files)}: {ndpi_file}\n")
        print(f"Level 0 dimensions: {level_0_dimensions}\n")
        print(f"Detection level {detection_level} dimensions: {level_dimensions}\n")
        print(f"Downsample factor: {downsample_factor:.2f}\n")
        print(f"Estimated magnification: {magnification}\n")
        print(f"Found {len(outer_contours)} tissue regions (green)\n")
        print(f"Found {len(hole_contours)} holes (blue)\n")
        print("===========================================")

        slide.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process ndpi/svs/czi files and generate masks and contours.")
    parser.add_argument("-pic_path", required=True, help="Path to the folder containing ndpi/svs/czi files.")
    parser.add_argument("-min_area", type=float, default=1000, help="Minimum area for tissue regions.")
    parser.add_argument("-min_hole", type=float, default=60000, help="Minimum area for holes.")
    parser.add_argument("-file_type", choices=["ndpi", "svs", "czi"], default="ndpi", help="Type of files to process: ndpi, svs, or czi.")
    parser.add_argument("-czi_level", type=int, default=6, help="CZI preview level. zoom = 1 / (2 ** czi_level).")
    parser.add_argument("-black_threshold", type=int, default=10, help="CZI pixels with all RGB channels below this value are treated as black regions.")
    parser.add_argument("-min_black_area", type=int, default=100, help="Minimum black connected component area to turn white for CZI files.")
    parser.add_argument("-gray_min_value", type=int, default=130, help="Minimum brightness for low-saturation gray background cleanup.")
    parser.add_argument("-gray_max_saturation", type=int, default=55, help="Maximum HSV saturation for gray background cleanup.")
    parser.add_argument("-pen_min_pixels", type=int, default=100, help="Minimum connected pixels for black pen marks to be removed. Set <= 0 to disable.")
    parser.add_argument("-pen_black_threshold", type=int, default=110, help="RGB threshold for black pen marks; pixels with all channels below this are candidates.")
    parser.add_argument("-kernel_size", type=int, default=20, help="Dilation kernel size for pen mark removal.")
    parser.add_argument("-dilate_iterations", type=int, default=1, help="Dilation iterations for pen mark removal.")
    parser.add_argument("-slice_brightness", type=int, default=20, help="Brightness added to CZI display images before pen mark removal.")
    parser.add_argument("-min_fragment_ratio", type=float, default=0.03, help="Minimum tissue fragment area as a ratio of the largest tissue region.")
    parser.add_argument("-overwrite", action="store_true", help="Overwrite existing output files. By default existing outputs are skipped.")
    parser.add_argument(
        "-out_dir",
        type=str,
        default=".",
        help="Output root directory (will create origin_pic/circle_pic/mask_pic under it; default: current dir).",
    )

    args = parser.parse_args()

    # 支持 ndpi 和 svs 文件
    def process_files(input_folder, min_area, min_hole_area, file_type, out_dir, czi_level, black_threshold, min_black_area, gray_min_value, gray_max_saturation, pen_min_pixels, pen_black_threshold, kernel_size, dilate_iterations, slice_brightness, min_fragment_ratio, overwrite):
        if file_type == "ndpi":
            file_ext = ".ndpi"
        elif file_type == "svs":
            file_ext = ".svs"
        else:
            file_ext = ".czi"
        files = [f for f in os.listdir(input_folder) if f.endswith(file_ext)]
        failed_files = []
        skipped_files = []

        for idx, file in enumerate(files, start=1):
            slide_path = os.path.join(input_folder, file)
            file_prefix = os.path.splitext(file)[0]

            origin_dir = os.path.join(out_dir, "origin_pic")
            circle_dir = os.path.join(out_dir, "circle_pic")
            mask_dir = os.path.join(out_dir, "mask_pic")
            os.makedirs(origin_dir, exist_ok=True)
            os.makedirs(circle_dir, exist_ok=True)
            os.makedirs(mask_dir, exist_ok=True)

            circle_path = os.path.join(circle_dir, f"{file_prefix}_tissue_contour_800x800.png")
            origin_path = os.path.join(origin_dir, f"{file_prefix}_thumbnail_800x800.png")
            mask_path = os.path.join(mask_dir, f"{file_prefix}_binary_mask_800x800.png")

            existing_outputs = [
                path for path in (circle_path, origin_path, mask_path)
                if os.path.exists(path)
            ]
            if existing_outputs and not overwrite:
                skipped_files.append(file)
                print(f"Skipping existing output {idx}/{len(files)}: {file}")
                continue

            slide = None
            if file_type == "czi":
                try:
                    detection_array, czi_zoom = read_czi_rgb(slide_path, czi_level)
                except Exception as exc:
                    failed_files.append((file, str(exc)))
                    print(f"Skipping unreadable CZI {idx}/{len(files)}: {file}")
                    print(f"   {exc}")
                    print("===========================================")
                    continue
                detection_level = czi_level
                level_dimensions = (detection_array.shape[1], detection_array.shape[0])
                downsample_factor = 1 / czi_zoom
                level_0_dimensions = (
                    int(level_dimensions[0] * downsample_factor),
                    int(level_dimensions[1] * downsample_factor),
                )
                magnification = "Unknown"
            else:
                try:
                    slide = openslide.OpenSlide(slide_path)
                except Exception as exc:
                    failed_files.append((file, str(exc)))
                    print(f"Skipping unreadable slide {idx}/{len(files)}: {file}")
                    print(f"   {exc}")
                    print("===========================================")
                    continue
                level_0_dimensions = slide.level_dimensions[0]
                mpp_x = slide.properties.get('openslide.mpp-x')
                mpp_y = slide.properties.get('openslide.mpp-y')
                if mpp_x and mpp_y:
                    avg_mpp = (float(mpp_x) + float(mpp_y)) / 2
                    magnification = "40x" if avg_mpp < 0.3 else "20x" if avg_mpp < 0.6 else "Unknown"
                else:
                    magnification = "Unknown"

            if file_type != "czi":
                detection_level = min(3, slide.level_count - 1)
                level_dimensions = slide.level_dimensions[detection_level]
                detection_img = slide.read_region((0, 0), detection_level, level_dimensions)
                detection_img = detection_img.convert('RGB')
                detection_array = np.array(detection_img)
                downsample_factor = slide.level_downsamples[detection_level]
            
            # 保存原始图像用于origin_pic
            if file_type == "czi":
                detection_array_raw = detection_array.copy()
                detection_array = clean_czi_preview(
                    detection_array,
                    black_threshold,
                    min_black_area,
                    gray_min_value,
                    gray_max_saturation,
                )
            else:
                detection_array_raw = detection_array.copy()

            if file_type == "czi":
                mask_detection_array = remove_pen_marks_fast(
                    detection_array,
                    pen_min_pixels,
                    pen_black_threshold,
                    kernel_size,
                    dilate_iterations,
                )
                result_detection = brighten_image(mask_detection_array, slice_brightness)
            else:
                mask_detection_array = detection_array
                result_detection = detection_array.copy()

            binary_opened, outer_contours, hole_contours = find_tissue_and_hole_contours(
                mask_detection_array,
                min_area,
                min_hole_area,
                downsample_factor,
                min_fragment_ratio,
            )
            result_detection_bgr = cv2.cvtColor(result_detection, cv2.COLOR_RGB2BGR)
            cv2.drawContours(result_detection_bgr, outer_contours, -1, (0, 255, 0), 3)
            cv2.drawContours(result_detection_bgr, hole_contours, -1, (255, 0, 0), 3)
            result_detection_rgb = cv2.cvtColor(result_detection_bgr, cv2.COLOR_BGR2RGB)
            h, w = result_detection_rgb.shape[:2]
            scale = min(800 / w, 800 / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            result_resized = cv2.resize(result_detection_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            if file_type == "czi":
                result_resized = clean_light_gray_background(
                    result_resized,
                    gray_min_value,
                    gray_max_saturation,
                )
            result_final = np.ones((800, 800, 3), dtype=np.uint8) * 255
            y_offset = (800 - new_h) // 2
            x_offset = (800 - new_w) // 2
            result_final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = result_resized
            result_pil = Image.fromarray(result_final)
            result_pil.save(circle_path)
            thumbnail_resized = cv2.resize(detection_array_raw, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            thumbnail_final = np.ones((800, 800, 3), dtype=np.uint8) * 255
            thumbnail_final[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = thumbnail_resized
            thumbnail_pil = Image.fromarray(thumbnail_final)
            thumbnail_pil.save(origin_path)
            mask_with_holes = np.zeros_like(binary_opened)
            cv2.drawContours(mask_with_holes, outer_contours, -1, 255, -1)
            cv2.drawContours(mask_with_holes, hole_contours, -1, 0, -1)
            binary_pil = Image.fromarray(mask_with_holes)
            binary_pil.save(mask_path)
            print(f"Processing file {idx}/{len(files)}: {file}\n")
            print(f"Level 0 dimensions: {level_0_dimensions}\n")
            print(f"Detection level {detection_level} dimensions: {level_dimensions}\n")
            print(f"Downsample factor: {downsample_factor:.2f}\n")
            print(f"Estimated magnification: {magnification}\n")
            print(f"Found {len(outer_contours)} tissue regions (green)\n")
            print(f"Found {len(hole_contours)} holes (blue)\n")
            print("===========================================")
            if slide is not None:
                slide.close()

        print("Batch summary")
        print(f"Processed: {len(files) - len(skipped_files) - len(failed_files)}")
        print(f"Skipped existing outputs: {len(skipped_files)}")
        print(f"Failed unreadable files: {len(failed_files)}")
        if skipped_files:
            print("Skipped files:")
            for file in skipped_files:
                print(f"  - {file}")
        if failed_files:
            print("Failed files:")
            for file, reason in failed_files:
                print(f"  - {file}: {reason}")

    process_files(
        args.pic_path,
        args.min_area,
        args.min_hole,
        args.file_type,
        args.out_dir,
        args.czi_level,
        args.black_threshold,
        args.min_black_area,
        args.gray_min_value,
        args.gray_max_saturation,
        args.pen_min_pixels,
        args.pen_black_threshold,
        args.kernel_size,
        args.dilate_iterations,
        args.slice_brightness,
        args.min_fragment_ratio,
        args.overwrite,
    )
