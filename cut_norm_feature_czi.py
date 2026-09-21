import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import openslide
import argparse
import torch
from torchvision import transforms
import h5py
from tqdm import tqdm
from queue import Queue
from threading import Thread, Lock

from conch.open_clip_custom import create_model_from_pretrained

CONCH_MEAN = (0.48145466, 0.4578275, 0.40821073)
CONCH_STD = (0.26862954, 0.26130258, 0.27577711)

try:
    from pylibCZIrw import czi
    HAS_CZI = True
except ImportError:
    HAS_CZI = False

# 导入标准化工具
try:
    from wsi_normalizer import imread, MacenkoNormalizer
    HAS_MACENKO = True
except ImportError:
    HAS_MACENKO = False
    print("Warning: wsi-normalizer not installed. Macenko method will not be available.")

try:
    import torchstain
    HAS_TORCHSTAIN = True
except ImportError:
    HAS_TORCHSTAIN = False
    print("Warning: torchstain not installed. GPU Macenko will not be available.")

def extract_features_adaptive(model, batch_tensor, initial_chunk_size=None):
    """Run CONCH feature extraction, splitting batches on CUDA capacity errors."""
    try:
        return model.encode_image(batch_tensor, proj_contrast=True, normalize=True)
    except RuntimeError as e:
        msg = str(e).lower()
        is_cuda_capacity_error = (
            "out of memory" in msg
            or "unable to find a valid cudnn algorithm" in msg
            or "cudnn" in msg
        )
        if not is_cuda_capacity_error or batch_tensor.shape[0] <= 1:
            raise

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        chunk_size = initial_chunk_size or max(1, batch_tensor.shape[0] // 2)
        chunk_size = min(chunk_size, max(1, batch_tensor.shape[0] // 2))
        print(f"⚠️  CONCH batch forward failed ({e}). Retrying with chunks of {chunk_size}.")

        outputs = []
        start = 0
        while start < batch_tensor.shape[0]:
            end = min(start + chunk_size, batch_tensor.shape[0])
            try:
                outputs.append(model.encode_image(
                    batch_tensor[start:end], proj_contrast=True, normalize=True
                ))
                start = end
            except RuntimeError as chunk_error:
                chunk_msg = str(chunk_error).lower()
                if (
                    ("out of memory" not in chunk_msg)
                    and ("unable to find a valid cudnn algorithm" not in chunk_msg)
                    and ("cudnn" not in chunk_msg)
                ) or chunk_size <= 1:
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                chunk_size = max(1, chunk_size // 2)
                print(f"⚠️  Chunk forward failed. Reducing CONCH chunk size to {chunk_size}.")

        return torch.cat(outputs, dim=0)

# ======================== GPU批量Macenko标准化 ========================
class GPUMacenkoBatchNormalizer:
    """GPU批量Macenko标准化器 - 关键性能优化"""
    def __init__(self, target_img_path, device, target_size=224):
        self.device = device
        self.target_size = int(target_size)
        
        # 读取参考图像
        target = cv2.imread(target_img_path)
        if target is None:
            raise FileNotFoundError(f"Cannot read reference image: {target_img_path}")
        
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # IMPORTANT:
        # torchstain 的 Macenko 在 torch<=1.13 的 GPU 路径中会对像素展开后做一次超大规模的 lstsq。
        # 当参考图过大时（像素数很大），torch.linalg.lstsq 可能触发 CUBLAS_STATUS_EXECUTION_FAILED。
        # Resize the reference to the CONCH patch size before fitting.
        if target.shape[0] != self.target_size or target.shape[1] != self.target_size:
            target = cv2.resize(target, (self.target_size, self.target_size), interpolation=cv2.INTER_AREA)
        
        # 转换为 Tensor (H,W,C) -> (C,H,W)
        target_tensor = torch.from_numpy(target).permute(2, 0, 1).float().to(device)
        
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
        
        try:
            # 拟合参考图像
            self.normalizer.fit(target_tensor)
            print(f"✅ GPU Macenko normalizer fitted successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to fit Macenko normalizer: {e}")
    
    def normalize_batch(self, batch_images):
        """
        批量标准化 - 核心优化点
        batch_images: numpy array (B, H, W, 3), uint8, range 0-255
        Returns: numpy array (B, H, W, 3), uint8
        """
        try:
            # 转为tensor (B, H, W, 3) -> (B, 3, H, W)
            batch_tensor = torch.from_numpy(batch_images).permute(0, 3, 1, 2).float().to(self.device)
            
            # torchstain需要范围0-255且逐个处理
            normalized_list = []
            for i in range(batch_tensor.shape[0]):
                single_img = batch_tensor[i]  # (3, H, W)
                result = self.normalizer.normalize(I=single_img, stains=False)
                
                # 处理可能的tuple返回值
                if isinstance(result, tuple):
                    norm_img = result[0]
                else:
                    norm_img = result
                
                # torchstain返回的是(H, W, 3)格式，需要转换为(3, H, W)
                if norm_img.shape[0] != 3:  # 如果不是(3, H, W)
                    norm_img = norm_img.permute(2, 0, 1)  # (H, W, 3) -> (3, H, W)
                
                normalized_list.append(norm_img)
            
            # 堆叠并转为numpy
            normalized_tensor = torch.stack(normalized_list, dim=0)  # (B, 3, H, W)
            # (B, 3, H, W) -> (B, H, W, 3)
            normalized_np = normalized_tensor.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            
            return normalized_np
            
        except Exception as e:
            print(f"Warning: GPU batch normalization failed: {e}, returning originals")
            return batch_images

# ======================== 模型加载 ========================
def get_conch_model(checkpoint_path):
    """Load the same CONCH model used for non-CZI slides."""
    print(f"📦 Loading CONCH model from: {checkpoint_path}")
    model, _ = create_model_from_pretrained(
        "conch_ViT-B-16", checkpoint_path=checkpoint_path
    )
    model.eval()
    return model

# ======================== 颜色标准化 ========================
def transform_to_lab(img):
    """转换为LAB色彩空间"""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return lab.astype(np.float64)

def reinhard_normalize(source_img, target_stats):
    """Reinhard颜色标准化"""
    target_mean, target_std = target_stats
    source_lab = transform_to_lab(source_img)
    L_source, A_source, B_source = cv2.split(source_lab)
    
    source_mean = np.array([L_source.mean(), A_source.mean(), B_source.mean()])
    source_std = np.array([L_source.std(), A_source.std(), B_source.std()])
    
    L_norm = ((L_source - source_mean[0]) * (target_std[0] / source_std[0])) + target_mean[0]
    A_norm = ((A_source - source_mean[1]) * (target_std[1] / source_std[1])) + target_mean[1]
    B_norm = ((B_source - source_mean[2]) * (target_std[2] / source_std[2])) + target_mean[2]
    
    L_norm = np.clip(L_norm, 0, 255)
    A_norm = np.clip(A_norm, 0, 255)
    B_norm = np.clip(B_norm, 0, 255)
    
    normalized_lab = cv2.merge([L_norm, A_norm, B_norm])
    normalized_bgr = cv2.cvtColor(normalized_lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    return normalized_bgr

def calculate_target_stats(target_img_path):
    """计算参考图像的LAB统计量"""
    target_img = cv2.imread(target_img_path)
    if target_img is None:
        raise FileNotFoundError(f"无法读取参考图像: {target_img_path}")
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    target_lab = transform_to_lab(target_img)
    L_target, A_target, B_target = cv2.split(target_lab)
    target_mean = np.array([L_target.mean(), A_target.mean(), B_target.mean()])
    target_std = np.array([L_target.std(), A_target.std(), B_target.std()])
    return target_mean, target_std

def prepare_normalizer(method, ref_img_path, device, patch_size):
    """准备颜色标准化器（优先GPU批量版本）"""
    if method == 'none':
        print("Color normalization disabled (-m none)")
        return None, None
    if method == 'macenko':
        # 优先使用GPU批量Macenko
        if HAS_TORCHSTAIN and torch.cuda.is_available():
            print(f"🚀 Using GPU batch Macenko normalizer (FAST!)")
            gpu_normalizer = GPUMacenkoBatchNormalizer(ref_img_path, device, target_size=patch_size)
            return gpu_normalizer, None
        elif HAS_MACENKO:
            print(f"⚠️  Falling back to CPU Macenko (SLOW)")
            normalizer = MacenkoNormalizer()
            normalizer.fit(imread(ref_img_path))
            return normalizer, None
        else:
            raise ImportError("Macenko method requires either torchstain (GPU) or wsi-normalizer (CPU)")
    else:  # reinhard
        print(f"Calculating Reinhard target stats from: {os.path.basename(ref_img_path)}")
        target_stats = calculate_target_stats(ref_img_path)
        print(f"Target stats - Mean: {target_stats[0]}, Std: {target_stats[1]}")
        return None, target_stats

def normalize_patch(patch_rgb, method, macenko_normalizer, target_stats):
    """标准化单个patch"""
    try:
        if method == 'none':
            return patch_rgb
        if method == 'macenko':
            # Macenko需要RGB输入
            normalized = macenko_normalizer.transform(patch_rgb)
            return normalized
        else:  # reinhard
            # Reinhard使用BGR
            normalized = reinhard_normalize(patch_rgb, target_stats)
            return normalized
    except Exception as e:
        print(f"Warning: Normalization failed: {e}")
        return patch_rgb

# ======================== Patch质量检测 ========================
def is_blank_patch(patch, threshold=0.8):
    """
    快速空白检测。
    亮白背景用亮度比例过滤；灰色/阴影背景用低饱和度 + 低纹理过滤。
    """
    patch_arr = np.array(patch)
    gray_patch = cv2.cvtColor(patch_arr, cv2.COLOR_RGB2GRAY)
    
    bright_pixel_ratio = np.sum(gray_patch > 230) / gray_patch.size
    if bright_pixel_ratio > threshold:
        return True

    hsv_patch = cv2.cvtColor(patch_arr, cv2.COLOR_RGB2HSV)
    saturation = hsv_patch[:, :, 1]
    value = hsv_patch[:, :, 2]

    gray_background_ratio = np.mean((saturation < 35) & (value > 145))
    texture_score = cv2.Laplacian(gray_patch, cv2.CV_64F).var()

    if gray_background_ratio > threshold and texture_score < 25:
        return True

    if saturation.mean() < 30 and gray_patch.mean() > 120 and texture_score < 18:
        return True

    return False

# ======================== CZI读取适配 ========================
class CziSlideReader:
    """Small adapter that exposes the OpenSlide methods used by this script."""
    def __init__(self, slide_path):
        if not HAS_CZI:
            raise ImportError("Reading .czi files requires pylibCZIrw in this environment.")

        self.slide_path = slide_path
        self._context = czi.open_czi(slide_path)
        self._reader = self._context.__enter__()
        bbox = self._reader.total_bounding_box_no_pyramid
        self.x0 = bbox["X"][0]
        self.y0 = bbox["Y"][0]
        self.dimensions = (bbox["X"][1] - bbox["X"][0], bbox["Y"][1] - bbox["Y"][0])
        self.level_dimensions = [self.dimensions]
        self.properties = {}

    def read_region(self, location, level, size):
        if level != 0:
            raise ValueError("CZI patch extraction only supports level 0 reads.")

        x, y = location
        w, h = size
        patch = self._reader.read(
            roi=(self.x0 + int(x), self.y0 + int(y), int(w), int(h)),
            zoom=1.0,
            background_pixel=(1.0, 1.0, 1.0),
        )
        patch = np.asarray(patch)
        if patch.ndim == 2:
            patch = np.stack([patch, patch, patch], axis=-1)
        if patch.shape[-1] > 3:
            patch = patch[..., :3]

        # pylibCZIrw returns BGR in this workflow, matching process_czi.py.
        return Image.fromarray(patch[..., ::-1].copy()).convert("RGB")

    def get_thumbnail(self, size):
        max_w, max_h = size
        zoom = min(max_w / self.dimensions[0], max_h / self.dimensions[1], 1.0)
        thumb = self._reader.read(
            zoom=zoom,
            background_pixel=(1.0, 1.0, 1.0),
        )
        thumb = np.asarray(thumb)
        if thumb.ndim == 2:
            thumb = np.stack([thumb, thumb, thumb], axis=-1)
        if thumb.shape[-1] > 3:
            thumb = thumb[..., :3]
        return Image.fromarray(thumb[..., ::-1].copy()).convert("RGB")

    def close(self):
        self._context.__exit__(None, None, None)

def open_slide_reader(slide_path, file_type):
    if file_type == "czi":
        return CziSlideReader(slide_path)
    return openslide.OpenSlide(slide_path)

def process_slide_wrapper(args_tuple):
    """
    Wrapper function for parallel processing of slides
    """
    (slide_path, mask_path, checkpoint_path, method, ref_img_path,
     patch_size, read_size, blank_threshold, batch_size, output_folder) = args_tuple
    
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    
    try:
        # 每个进程需要重新加载模型
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        # 加载模型
        model = get_conch_model(checkpoint_path)
        model = model.to(device)
        
        # transform 不再做 resize：patch 会在 read_size 后先 resize 到 patch_size
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CONCH_MEAN, std=CONCH_STD),
        ])
        
        # 准备normalizer
        macenko_normalizer, target_stats = prepare_normalizer(method, ref_img_path, device, patch_size)
        
        # 处理切片
        result = process_single_slide(
            slide_path=slide_path,
            mask_path=mask_path,
            model=model,
            transform=transform,
            device=device,
            method=method,
            macenko_normalizer=macenko_normalizer,
            target_stats=target_stats,
            patch_size=patch_size,
            read_size=read_size,
            blank_threshold=blank_threshold,
            batch_size=batch_size,
            output_folder=output_folder,
            save_raw_patches=None,
            save_normalized_patches=None
        )
        
        if result is not None:
            # 保存h5文件
            h5_path = os.path.join(output_folder, f"{result['slide_name']}.h5")
            save_to_h5(result, h5_path)
            return True, slide_name, None
        else:
            return False, slide_name, "No valid patches found"
            
    except Exception as e:
        return False, slide_name, str(e)

# ======================== 多线程预加载 ========================
class PatchLoader:
    """多线程预加载patches，避免GPU等待IO"""
    def __init__(self, slide, positions, read_size_val, patch_size, blank_threshold, 
                 method, macenko_normalizer, target_stats, transform, num_workers=4, queue_size=10):
        self.slide = slide
        self.positions = positions
        self.read_size_val = read_size_val
        self.patch_size = patch_size
        self.blank_threshold = blank_threshold
        self.method = method
        self.macenko_normalizer = macenko_normalizer
        self.target_stats = target_stats
        self.transform = transform
        self.num_workers = num_workers
        self.lock = Lock()
        self.queue = Queue(maxsize=queue_size)
        self.workers = []
        self.current_idx = 0
        self.stop_flag = False
        
    def _worker(self):
        """Worker线程：加载和预处理patches"""
        while not self.stop_flag:
            # 获取下一个要处理的位置
            with self.lock:
                if self.current_idx >= len(self.positions):
                    break
                idx = self.current_idx
                self.current_idx += 1
            
            pos_data = self.positions[idx]
            
            try:
                # 读取patch
                region = self.slide.read_region((pos_data['x'], pos_data['y']), 0, 
                                               (self.read_size_val, self.read_size_val))
                region = region.convert("RGB")
                
                # Resize if needed
                if self.read_size_val != self.patch_size:
                    region = region.resize((self.patch_size, self.patch_size), Image.BILINEAR)
                
                # 检查是否为空白
                if is_blank_patch(region, self.blank_threshold):
                    self.queue.put(None)  # 标记为跳过
                    continue
                
                # 标准化
                patch_rgb = np.array(region)
                normalized_patch = normalize_patch(patch_rgb, self.method, 
                                                   self.macenko_normalizer, self.target_stats)
                
                # 转换为tensor
                pil_image = Image.fromarray(normalized_patch)
                tensor = self.transform(pil_image)
                
                # 放入队列
                self.queue.put({
                    'tensor': tensor,
                    'coord': [pos_data['x'], pos_data['y']],
                    'index': pos_data['index']
                })
                
            except Exception as e:
                print(f"\nError loading patch at ({pos_data['x']}, {pos_data['y']}): {e}")
                self.queue.put(None)
    
    def start(self):
        """启动worker线程"""
        from threading import Lock
        self.lock = Lock()
        self.current_idx = 0
        
        for _ in range(self.num_workers):
            worker = Thread(target=self._worker)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def get_batch(self, batch_size):
        """获取一个batch的数据"""
        batch_tensors = []
        batch_coords = []
        batch_indices = []
        
        for _ in range(batch_size):
            data = self.queue.get()
            if data is None:  # 跳过的patch
                continue
            if data == 'DONE':  # 结束标记
                break
            
            batch_tensors.append(data['tensor'])
            batch_coords.append(data['coord'])
            batch_indices.append(data['index'])
        
        return batch_tensors, batch_coords, batch_indices
    
    def stop(self):
        """停止所有worker"""
        self.stop_flag = True
        for worker in self.workers:
            worker.join(timeout=1)

# 全局变量，用于在worker进程中缓存OpenSlide对象
global_slide = None
global_slide_path = None
global_slide_file_type = None

def iter_contiguous_row_groups(batch_positions, read_size, max_group_patches):
    indexed_positions = [
        (order_idx, pos_data)
        for order_idx, pos_data in enumerate(batch_positions)
    ]
    indexed_positions.sort(key=lambda item: (item[1]['y'], item[1]['x']))

    groups = []
    current = []
    last_x = None
    last_y = None
    max_group_patches = max(1, int(max_group_patches))

    for item in indexed_positions:
        _, pos_data = item
        same_row = last_y == pos_data['y']
        next_patch = last_x is not None and pos_data['x'] == last_x + read_size
        has_room = len(current) < max_group_patches

        if current and same_row and next_patch and has_room:
            current.append(item)
        else:
            if current:
                groups.append(current)
            current = [item]

        last_x = pos_data['x']
        last_y = pos_data['y']

    if current:
        groups.append(current)
    return groups

def load_patch_from_region(region, pos_data, patch_size, blank_threshold):
    region = region.convert("RGB")
    region = region.resize((patch_size, patch_size), Image.BILINEAR)
    if is_blank_patch(region, blank_threshold):
        return None
    return np.array(region)

def load_czi_patch_group(slide, group, read_size, patch_size, blank_threshold):
    y = group[0][1]['y']
    left = group[0][1]['x']
    right = group[-1][1]['x'] + read_size
    strip_width = right - left

    strip = slide.read_region((left, y), 0, (strip_width, read_size)).convert("RGB")
    loaded = []
    filtered = []

    for order_idx, pos_data in group:
        offset_x = pos_data['x'] - left
        patch = strip.crop((offset_x, 0, offset_x + read_size, read_size))
        patch_rgb = load_patch_from_region(patch, pos_data, patch_size, blank_threshold)
        if patch_rgb is None:
            filtered.append((order_idx, pos_data))
        else:
            loaded.append((order_idx, pos_data, patch_rgb))

    return loaded, filtered

# ======================== Worker函数（必须在顶层才能被pickle） ========================
def process_batch_worker(batch_data):
    """Worker进程：加载patches（不做标准化，留给GPU批处理）"""
    global global_slide, global_slide_path, global_slide_file_type
    
    (batch_idx, batch_positions, slide_path_local, file_type_local, read_size_val_local,
     patch_size_local, blank_threshold_local, czi_strip_patches_local) = batch_data
    
    try:
        # 懒加载OpenSlide对象，避免每个batch重复打开/关闭
        if global_slide is None or global_slide_path != slide_path_local or global_slide_file_type != file_type_local:
            if global_slide is not None:
                global_slide.close()
            global_slide = open_slide_reader(slide_path_local, file_type_local)
            global_slide_path = slide_path_local
            global_slide_file_type = file_type_local
        
        slide_local = global_slide
        
        batch_images = []  # 存储原始numpy图像
        batch_coords = []
        batch_indices = []
        batch_kept_pos = []
        batch_filtered_pos = []

        loaded_items = []
        if file_type_local == "czi" and czi_strip_patches_local > 1:
            groups = iter_contiguous_row_groups(
                batch_positions,
                read_size_val_local,
                czi_strip_patches_local,
            )
            for group in groups:
                try:
                    loaded, filtered = load_czi_patch_group(
                        slide_local,
                        group,
                        read_size_val_local,
                        patch_size_local,
                        blank_threshold_local,
                    )
                    loaded_items.extend(loaded)
                    batch_filtered_pos.extend([
                        (pos_data['x'], pos_data['y'])
                        for _, pos_data in filtered
                    ])
                except Exception:
                    for order_idx, pos_data in group:
                        try:
                            region = slide_local.read_region(
                                (pos_data['x'], pos_data['y']),
                                0,
                                (read_size_val_local, read_size_val_local),
                            )
                            patch_rgb = load_patch_from_region(
                                region,
                                pos_data,
                                patch_size_local,
                                blank_threshold_local,
                            )
                            if patch_rgb is None:
                                batch_filtered_pos.append((pos_data['x'], pos_data['y']))
                            else:
                                loaded_items.append((order_idx, pos_data, patch_rgb))
                        except Exception:
                            batch_filtered_pos.append((pos_data['x'], pos_data['y']))
        else:
            for order_idx, pos_data in enumerate(batch_positions):
                try:
                    region = slide_local.read_region((pos_data['x'], pos_data['y']), 0, 
                                                     (read_size_val_local, read_size_val_local))
                    patch_rgb = load_patch_from_region(
                        region,
                        pos_data,
                        patch_size_local,
                        blank_threshold_local,
                    )
                    if patch_rgb is None:
                        batch_filtered_pos.append((pos_data['x'], pos_data['y']))
                    else:
                        loaded_items.append((order_idx, pos_data, patch_rgb))
                except Exception:
                    batch_filtered_pos.append((pos_data['x'], pos_data['y']))

        loaded_items.sort(key=lambda item: item[0])
        for _, pos_data, patch_rgb in loaded_items:
            batch_images.append(patch_rgb)
            batch_coords.append([pos_data['x'], pos_data['y']])
            batch_indices.append(pos_data['index'])
            batch_kept_pos.append((pos_data['x'], pos_data['y']))
        
        # 注意：不要关闭slide_local，因为它是全局缓存的
        # slide_local.close()
        
        # 返回原始图像数组（不是tensor）
        return {
            'batch_idx': batch_idx,
            'images': batch_images,  # numpy数组列表，不是tensor
            'coords': batch_coords,
            'indices': batch_indices,
            'kept_pos': batch_kept_pos,
            'filtered_pos': batch_filtered_pos,
            'total_in_batch': len(batch_positions)
        }
    except Exception as e:
        print(f"[Worker] CRITICAL ERROR in Batch {batch_idx}: {e}", flush=True)
        return None

# ======================== 主处理流程 ========================
def process_single_slide(slide_path, mask_path, model, transform, device, 
                        method, macenko_normalizer, target_stats,
                        patch_size, read_size, blank_threshold, batch_size,
                        output_folder, checkpoint_interval=50, save_filter_log=True,
                        grid_output_dir=None, num_workers=4, ref_img_path=None,
                        save_raw_patches=None, save_normalized_patches=None,
                        file_type="svs", czi_strip_patches=16,
                        max_saved_patches=None, min_mask_ratio=0.0):
    """
    处理单个切片：切patch -> 标准化 -> 特征提取
    支持增量保存和断点续传（每N个batch保存一次）
    """
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    h5_path = os.path.join(output_folder, f"{slide_name}.h5")
    filter_log_path = os.path.join(output_folder, f"{slide_name}_filter_log.txt")
    
    # 创建patch保存目录（如果需要）
    raw_patch_dir = None
    normalized_patch_dir = None
    if save_raw_patches:
        raw_patch_dir = os.path.join(save_raw_patches, slide_name)
        os.makedirs(raw_patch_dir, exist_ok=True)
        print(f"📁 Raw patches will be saved to: {raw_patch_dir}")
    if save_normalized_patches:
        normalized_patch_dir = os.path.join(save_normalized_patches, slide_name)
        os.makedirs(normalized_patch_dir, exist_ok=True)
        print(f"📁 Normalized patches will be saved to: {normalized_patch_dir}")
    raw_saved_count = 0
    normalized_saved_count = 0
    if max_saved_patches is not None and max_saved_patches <= 0:
        max_saved_patches = None
    if max_saved_patches is not None and (raw_patch_dir is not None or normalized_patch_dir is not None):
        print(f"📌 Saving at most {max_saved_patches} patches per saved-patch folder for this slide")
    
    print(f"\n{'='*60}")
    print(f"Processing slide: {slide_name}")
    print(f"{'='*60}")
    
    # 检查是否有已处理的数据
    existing_data, processed_indices = load_existing_h5(h5_path)
    
    # 准备过滤日志和可视化数据
    filtered_patches = []
    kept_patch_positions = []
    filtered_patch_positions = []
    
    # 1. 打开切片和mask
    try:
        slide = open_slide_reader(slide_path, file_type)
    except Exception as e:
        print(f"Error opening slide: {e}")
        return None
    
    try:
        mask = Image.open(mask_path)
    except FileNotFoundError:
        print(f"Error: Mask not found at {mask_path}")
        slide.close()
        return None
    
    mask = np.array(mask)
    level_0_dimensions = slide.level_dimensions[0]
    level_0_width, level_0_height = level_0_dimensions
    mask_height, mask_width = mask.shape[:2]
    
    # 2. 检测magnification
    mpp_x = slide.properties.get('openslide.mpp-x')
    mpp_y = slide.properties.get('openslide.mpp-y')
    if mpp_x and mpp_y:
        avg_mpp = (float(mpp_x) + float(mpp_y)) / 2
        if avg_mpp < 0.3:
            magnification = "40x"
        elif avg_mpp < 0.6:
            magnification = "20x"
        else:
            magnification = "Unknown"
    else:
        magnification = "Unknown"
    
    print(f"Magnification: {magnification} (avg_mpp: {avg_mpp if mpp_x else 'N/A'})")
    if file_type == "czi":
        print("CZI note: read_size and saved coordinates are level-0 CZI pixels.")
    
    if read_size is None:
        read_size_val = 512 if magnification == "40x" else patch_size
    else:
        read_size_val = read_size
    
    step = read_size_val
    print(f"Parameters: read_size={read_size_val}, patch_size={patch_size}, step={step}")
    print(f"Mask candidate threshold: min_mask_ratio={min_mask_ratio}")
    
    # 保存level-0 shape用于后续可视化；不要把800x800 mask放大到level-0，
    # 大TCGA切片会产生数GB到十几GB数组，容易触发OOM。
    mask_shape = (level_0_height, level_0_width)
    
    # 3. 扫描所有有效位置
    print("\nStep 1/2: Scanning for valid patch positions...")
    valid_positions = []
    patch_counter = 0
    
    total_positions = ((level_0_height + step - 1) // step) * ((level_0_width + step - 1) // step)
    
    with tqdm(total=total_positions, desc="Scanning positions", unit="pos") as pbar:
        for y in range(0, level_0_height, step):
            for x in range(0, level_0_width, step):
                pbar.update(1)
                
                x0 = int(x * mask_width / level_0_width)
                x1 = int(min(x + step, level_0_width) * mask_width / level_0_width)
                y0 = int(y * mask_height / level_0_height)
                y1 = int(min(y + step, level_0_height) * mask_height / level_0_height)
                x1 = max(x1, x0 + 1)
                y1 = max(y1, y0 + 1)
                patch_mask = mask[y0:y1, x0:x1]
                mask_ratio = np.mean(patch_mask > 0)
                if mask_ratio <= min_mask_ratio:
                    continue
                
                # 跳过已处理的patch
                if patch_counter not in processed_indices:
                    valid_positions.append({
                        'x': x,
                        'y': y,
                        'index': patch_counter
                    })
                
                patch_counter += 1
    
    total_candidates = patch_counter
    already_processed = len(processed_indices)
    to_process = len(valid_positions)
    
    print(f"Total candidate positions: {total_candidates}")
    print(f"Already processed: {already_processed}")
    print(f"To process: {to_process}")
    
    if to_process == 0:
        print("✅ All patches already processed!")
        slide.close()
        
        # 即使已经全部处理完，也生成grid可视化
        if grid_output_dir is not None and existing_data is not None:
            os.makedirs(grid_output_dir, exist_ok=True)
            grid_path = os.path.join(grid_output_dir, f"{slide_name}_grid.png")
            
            print(f"\n📊 Generating grid visualization for completed slide...")
            
            # 从已保存的h5加载所有位置
            kept_positions_from_h5 = [(int(coord[0]), int(coord[1])) for coord in existing_data['coordinates']]
            
            print(f"   Kept patches: {len(kept_positions_from_h5)}")
            
            save_grid_visualization(
                slide_path=slide_path,
                file_type=file_type,
                mask_shape=mask_shape,
                step=step,
                kept_positions=kept_positions_from_h5,
                filtered_positions=[],
                output_path=grid_path
            )
        
        return {'slide_name': slide_name, 'skipped': True}
    
    # 4. 处理patches - 使用多进程并行加载
    print(f"\nStep 2/2: Processing remaining patches...")
    print(f"Batch size: {batch_size}, Checkpoint every {checkpoint_interval} batches")
    print(f"Using {num_workers} worker processes for parallel patch loading")
    if file_type == "czi" and czi_strip_patches > 1:
        print(f"CZI strip read optimization: up to {czi_strip_patches} contiguous patches per ROI")
    
    all_features = []
    all_coords = []
    all_indices = []
    
    num_batches = (to_process + batch_size - 1) // batch_size
    processed_count = 0
    
    # 使用多进程并行加载patches
    from multiprocessing import Pool, Manager
    
    # 构建坐标到索引的映射，避免后续O(n)查找
    coord_to_index = {(vp['x'], vp['y']): vp['index'] for vp in valid_positions}
    
    # 准备所有batch的数据（简化参数，标准化移到GPU批处理）
    batch_tasks = []
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, to_process)
        batch_positions = valid_positions[start_idx:end_idx]
        
        batch_tasks.append((
            batch_idx,
            batch_positions,
            slide_path,
            file_type,
            read_size_val,
            patch_size,
            blank_threshold,
            czi_strip_patches
        ))
    
    # 使用进程池并行处理
    # 使用imap_unordered提高并行效率，用字典缓存结果保持顺序
    with Pool(processes=num_workers) as pool:
        # 用于缓存乱序结果并重新排序
        pending_results = {}
        next_batch_to_process = 0
        
        with tqdm(total=to_process, desc="Processing patches", unit="patch") as pbar:
            # 使用imap_unordered允许快的batch先返回，避免被慢batch阻塞
            for batch_result in pool.imap_unordered(process_batch_worker, batch_tasks, chunksize=1):
                if batch_result is None:
                    continue
                
                # 缓存结果
                batch_idx = batch_result['batch_idx']
                pending_results[batch_idx] = batch_result
                
                # 按顺序处理已完成的batch
                while next_batch_to_process in pending_results:
                    result = pending_results.pop(next_batch_to_process)
                    
                    # 更新进度条
                    pbar.update(result['total_in_batch'])
                    
                    # 收集过滤结果
                    kept_patch_positions.extend(result['kept_pos'])
                    filtered_patch_positions.extend(result['filtered_pos'])
                    
                    # 保存过滤日志 (使用字典O(1)查找)
                    if save_filter_log:
                        for fpos in result['filtered_pos']:
                            idx = coord_to_index.get(fpos)
                            if idx is not None:
                                filtered_patches.append({
                                    'index': idx,
                                    'x': fpos[0],
                                    'y': fpos[1],
                                    'reason': 'blank or error'
                                })
                    
                    # GPU批量标准化 + 特征提取
                    if len(result['images']) > 0:
                        # Step 1: 批量标准化（如果需要）
                        batch_images_np = np.stack(result['images'], axis=0)  # (B, 224, 224, 3) uint8
                        
                        # 保存原始patches（如果需要）
                        if raw_patch_dir is not None:
                            for idx, (img, coord, patch_idx) in enumerate(zip(batch_images_np, result['coords'], result['indices'])):
                                if max_saved_patches is not None and raw_saved_count >= max_saved_patches:
                                    break
                                patch_filename = f"patch_{patch_idx:05d}_x{coord[0]}_y{coord[1]}.png"
                                patch_path = os.path.join(raw_patch_dir, patch_filename)
                                Image.fromarray(img).save(patch_path)
                                raw_saved_count += 1
                        
                        if method != 'none':
                            if isinstance(macenko_normalizer, GPUMacenkoBatchNormalizer):
                                # GPU批量Macenko标准化 - 关键优化！
                                normalized_batch = macenko_normalizer.normalize_batch(batch_images_np)
                            elif macenko_normalizer is not None:
                                # CPU Macenko（fallback，慢）
                                normalized_batch = np.stack([
                                    normalize_patch(img, method, macenko_normalizer, target_stats)
                                    for img in batch_images_np
                                ], axis=0)
                            elif target_stats is not None:
                                # Reinhard标准化
                                normalized_batch = np.stack([
                                    reinhard_normalize(img, target_stats)
                                    for img in batch_images_np
                                ], axis=0)
                            else:
                                normalized_batch = batch_images_np
                        else:
                            normalized_batch = batch_images_np
                        
                        # 保存标准化后的patches（如果需要）
                        if normalized_patch_dir is not None:
                            for idx, (img, coord, patch_idx) in enumerate(zip(normalized_batch, result['coords'], result['indices'])):
                                if max_saved_patches is not None and normalized_saved_count >= max_saved_patches:
                                    break
                                patch_filename = f"patch_{patch_idx:05d}_x{coord[0]}_y{coord[1]}.png"
                                patch_path = os.path.join(normalized_patch_dir, patch_filename)
                                Image.fromarray(img).save(patch_path)
                                normalized_saved_count += 1
                        
                        # Step 2: 转换为tensor并标准化
                        # (B, H, W, 3) -> (B, 3, H, W), float, 0-1
                        batch_tensor = torch.from_numpy(normalized_batch).permute(0, 3, 1, 2).float() / 255.0
                        
                        # CONCH/OpenCLIP normalization.
                        mean = torch.tensor(CONCH_MEAN, device=batch_tensor.device).view(1, 3, 1, 1)
                        std = torch.tensor(CONCH_STD, device=batch_tensor.device).view(1, 3, 1, 1)
                        batch_tensor = (batch_tensor - mean) / std
                        
                        # Step 3: GPU特征提取
                        batch_tensor = batch_tensor.to(device, non_blocking=True)
                        with torch.inference_mode():
                            batch_features = extract_features_adaptive(model, batch_tensor)
                        
                        batch_features = batch_features.cpu().numpy()
                        all_features.append(batch_features)
                        all_coords.extend(result['coords'])
                        all_indices.extend(result['indices'])
                        processed_count += len(result['images'])
                    
                    # 定期保存checkpoint
                    if (next_batch_to_process + 1) % checkpoint_interval == 0 and len(all_features) > 0:
                        features = np.vstack(all_features)
                        coordinates = np.array(all_coords, dtype=np.int64)
                        patch_indices = np.array(all_indices, dtype=np.int64)
                        
                        new_data = {
                            'features': features,
                            'coordinates': coordinates,
                            'patch_indices': patch_indices
                        }
                        
                        append_to_h5(h5_path, new_data, existing_data)
                        print(f"\n💾 Checkpoint saved: {len(features)} new patches (Total processed: {already_processed + processed_count})")
                        
                        # 优化：不再累积existing_data,避免O(n²)内存复制
                        # existing_data只在最开始用于断点续传判断,后续checkpoint不需要更新它
                        
                        # 清空临时数据
                        all_features = []
                        all_coords = []
                        all_indices = []
                    
                    next_batch_to_process += 1
    
    # 关闭主线程的slide（worker已经关闭了自己的）
    # slide.close()  # 不需要关闭,因为worker进程各自管理
    
    # 保存过滤日志
    if save_filter_log and len(filtered_patches) > 0:
        with open(filter_log_path, 'w') as f:
            f.write(f"Slide: {slide_name}\n")
            f.write(f"Total candidates: {total_candidates}\n")
            f.write(f"Filtered out: {len(filtered_patches)}\n")
            f.write(f"Valid patches: {processed_count}\n")
            f.write(f"Blank threshold: {blank_threshold}\n")
            f.write(f"\nFiltered patches:\n")
            f.write("Index, X, Y, Reason\n")
            for fp in filtered_patches:
                f.write(f"{fp['index']}, {fp['x']}, {fp['y']}, {fp['reason']}\n")
        print(f"📋 Filter log saved to: {filter_log_path}")
    
    # 最后保存剩余的数据
    if len(all_features) > 0:
        features = np.vstack(all_features)
        coordinates = np.array(all_coords, dtype=np.int64)
        patch_indices = np.array(all_indices, dtype=np.int64)
        
        new_data = {
            'features': features,
            'coordinates': coordinates,
            'patch_indices': patch_indices
        }
        
        append_to_h5(h5_path, new_data, existing_data)
        print(f"\n💾 Final save: {len(features)} patches")
    
    # 读取最终结果
    with h5py.File(h5_path, 'r') as f:
        total_patches = len(f['patch_indices'])
    
    print(f"\n✅ Slide completed: {total_patches} total patches")
    print(f"   ({already_processed} previous + {processed_count} new)")
    
    # 生成网格可视化
    if grid_output_dir is not None:
        os.makedirs(grid_output_dir, exist_ok=True)
        grid_path = os.path.join(grid_output_dir, f"{slide_name}_grid.png")
        
        print(f"\n📊 Generating grid visualization...")
        
        # 如果是断点续传，需要加载之前保留的位置
        if existing_data is not None and len(existing_data['coordinates']) > 0:
            for coord in existing_data['coordinates']:
                kept_patch_positions.append((int(coord[0]), int(coord[1])))
        
        print(f"   Kept patches: {len(kept_patch_positions)}")
        print(f"   Filtered patches: {len(filtered_patch_positions)}")
        
        save_grid_visualization(
            slide_path=slide_path,
            file_type=file_type,
            mask_shape=mask_shape,
            step=step,
            kept_positions=kept_patch_positions,
            filtered_positions=filtered_patch_positions,
            output_path=grid_path
        )
    
    return {'slide_name': slide_name, 'total_patches': total_patches}

def save_grid_visualization(slide_path, file_type, mask_shape, step, kept_positions, filtered_positions, output_path):
    """
    生成网格可视化图，在切片缩略图上叠加网格
    
    参数:
    - slide_path: 切片文件路径
    - mask_shape: (height, width) 原始mask的尺寸
    - step: patch的步长
    - kept_positions: 保留的patches的位置列表 [(x, y), ...]
    - filtered_positions: 被过滤的patches的位置列表 [(x, y), ...]
    - output_path: 保存路径
    """
    print(f"   Creating grid visualization with slide thumbnail...")
    
    try:
        # 限制缩略图最大边为1000像素，等比例缩放
        max_thumb = 1000
        padding = 50  # 添加边距
        
        scale = min(max_thumb / mask_shape[1], max_thumb / mask_shape[0], 1.0)
        thumb_w = int(mask_shape[1] * scale)
        thumb_h = int(mask_shape[0] * scale)
        
        # 打开切片获取缩略图
        slide = open_slide_reader(slide_path, file_type)
        thumbnail = slide.get_thumbnail((thumb_w, thumb_h)).convert('RGB')
        slide.close()
        
        # 创建带边距的画布，并将缩略图居中放置
        canvas_w = thumb_w + 2 * padding
        canvas_h = thumb_h + 2 * padding
        canvas = Image.new('RGB', (canvas_w, canvas_h), color=(255, 255, 255))
        
        # 将缩略图粘贴到画布中央
        offset_x = padding
        offset_y = padding
        canvas.paste(thumbnail, (offset_x, offset_y))
        
        print(f"   Thumbnail size: {thumb_w}x{thumb_h}, Canvas size: {canvas_w}x{canvas_h}, scale: {scale:.4f}")
        draw = ImageDraw.Draw(canvas, 'RGBA')
        
        # 绘制保留patch（手动绘制四条边，确保线条粗细一致）
        line_width = 2
        line_color = (0, 0, 0, 255)
        
        for x, y in kept_positions:
            # 应用缩放和偏移
            x1 = int(x * scale) + offset_x
            y1 = int(y * scale) + offset_y
            x2 = int((x + step) * scale) + offset_x
            y2 = int((y + step) * scale) + offset_y
            
            # 手动绘制四条边，线条居中在矩形边界上
            # 上边
            draw.line([(x1, y1), (x2, y1)], fill=line_color, width=line_width)
            # 下边
            draw.line([(x1, y2), (x2, y2)], fill=line_color, width=line_width)
            # 左边
            draw.line([(x1, y1), (x1, y2)], fill=line_color, width=line_width)
            # 右边
            draw.line([(x2, y1), (x2, y2)], fill=line_color, width=line_width)
        
        # 被过滤patch完全透明（不绘制）
        canvas.save(output_path)
        print(f"   ✅ Grid visualization saved successfully!")
    except Exception as e:
        print(f"   ❌ Error creating visualization: {e}")
        raise

def load_existing_h5(h5_path):
    """加载已有的h5文件，返回已处理的patch索引集合"""
    if not os.path.exists(h5_path):
        return None, set()
    
    try:
        with h5py.File(h5_path, 'r') as f:
            existing_indices = set(f['patch_indices'][:].tolist())
            existing_data = {
                'features': f['features'][:],
                'coordinates': f['coordinates'][:],
                'patch_indices': f['patch_indices'][:]
            }
        print(f"   📂 Found existing h5 with {len(existing_indices)} patches")
        return existing_data, existing_indices
    except Exception as e:
        print(f"   ⚠️  Error reading existing h5: {e}, will start fresh")
        return None, set()

def append_to_h5(h5_path, new_data, existing_data=None):
    """追加或创建h5文件 (优化版：支持自动迁移旧格式)"""
    if not os.path.exists(h5_path):
        # 创建新文件，启用chunked存储以支持resize
        with h5py.File(h5_path, 'w') as f:
            feature_dim = new_data['features'].shape[1]
            f.create_dataset('features', data=new_data['features'], compression='gzip', maxshape=(None, feature_dim), chunks=True)
            f.create_dataset('coordinates', data=new_data['coordinates'], compression='gzip', maxshape=(None, 2), chunks=True)
            f.create_dataset('patch_indices', data=new_data['patch_indices'], compression='gzip', maxshape=(None,), chunks=True)
    else:
        try:
            # 尝试追加到现有文件
            with h5py.File(h5_path, 'a') as f:
                # 检查是否支持resize (maxshape是否为None或足够大)
                if f['features'].maxshape[0] is not None:
                    # 如果maxshape是固定的，可能无法resize，抛出异常触发迁移
                    current_max = f['features'].maxshape[0]
                    new_needed = f['features'].shape[0] + new_data['features'].shape[0]
                    if new_needed > current_max:
                        raise RuntimeError(f"Dataset maxshape too small: {current_max} < {new_needed}")

                # 获取当前大小
                current_size = f['features'].shape[0]
                new_size = current_size + new_data['features'].shape[0]
                
                # 调整大小
                feature_dim = f['features'].shape[1]
                if new_data['features'].shape[1] != feature_dim:
                    raise ValueError(
                        f"Feature dimension changed from {feature_dim} to "
                        f"{new_data['features'].shape[1]}"
                    )
                f['features'].resize((new_size, feature_dim))
                f['coordinates'].resize((new_size, 2))
                f['patch_indices'].resize((new_size,))
                
                # 写入新数据
                f['features'][current_size:] = new_data['features']
                f['coordinates'][current_size:] = new_data['coordinates']
                f['patch_indices'][current_size:] = new_data['patch_indices']
                
        except (RuntimeError, OSError, ValueError) as e:
            print(f"\n⚠️  Error appending to H5 ({e}). Migrating to new format...")
            # 如果追加失败（通常是因为旧文件不支持resize），则重写整个文件
            # 从磁盘读取当前文件中的所有数据（不依赖existing_data参数）
            try:
                with h5py.File(h5_path, 'r') as f:
                    current_features = f['features'][:]
                    current_coords = f['coordinates'][:]
                    current_indices = f['patch_indices'][:]
                
                combined_features = np.vstack([current_features, new_data['features']])
                combined_coords = np.vstack([current_coords, new_data['coordinates']])
                combined_indices = np.concatenate([current_indices, new_data['patch_indices']])
                
            except Exception as read_err:
                print(f"❌ Critical error reading existing file: {read_err}")
                # 如果连读取都失败了，只能用new_data覆盖（极端情况）
                print("⚠️  WARNING: Overwriting with new data only. Previous data may be lost!")
                combined_features = new_data['features']
                combined_coords = new_data['coordinates']
                combined_indices = new_data['patch_indices']
            
            # 重写文件
            with h5py.File(h5_path, 'w') as f:
                feature_dim = combined_features.shape[1]
                f.create_dataset('features', data=combined_features, compression='gzip', maxshape=(None, feature_dim), chunks=True)
                f.create_dataset('coordinates', data=combined_coords, compression='gzip', maxshape=(None, 2), chunks=True)
                f.create_dataset('patch_indices', data=combined_indices, compression='gzip', maxshape=(None,), chunks=True)
            print("✅ Migration completed. File is now resizable.")

def save_to_h5(data, h5_path):
    """保存特征到h5文件"""
    with h5py.File(h5_path, 'w') as f:
        f.create_dataset('features', data=data['features'], compression='gzip')
        f.create_dataset('coordinates', data=data['coordinates'], compression='gzip')
        f.create_dataset('patch_indices', data=data['patch_indices'], compression='gzip')
    
    print(f"\n💾 Saved to: {h5_path}")
    print(f"   H5 Contents:")
    print(f"   - features: {data['features'].shape}, dtype: {data['features'].dtype}")
    print(f"   - coordinates: {data['coordinates'].shape}, dtype: {data['coordinates'].dtype}")
    print(f"   - patch_indices: {data['patch_indices'].shape}, dtype: {data['patch_indices'].dtype}")

def process_slide_wrapper(args_tuple):
    """
    Wrapper function for parallel processing of slides
    """
    (slide_path, mask_path, checkpoint_path, method, ref_img_path,
     patch_size, read_size, blank_threshold, batch_size, output_folder) = args_tuple
    
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    
    try:
        # 每个进程需要重新加载模型
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        # 加载模型
        model = get_conch_model(checkpoint_path)
        model = model.to(device)
        
        # transform 不再做 resize：patch 会在 read_size 后先 resize 到 patch_size
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CONCH_MEAN, std=CONCH_STD),
        ])
        
        # 准备normalizer
        macenko_normalizer, target_stats = prepare_normalizer(method, ref_img_path, device, patch_size)
        
        # 处理切片
        result = process_single_slide(
            slide_path=slide_path,
            mask_path=mask_path,
            model=model,
            transform=transform,
            device=device,
            method=method,
            macenko_normalizer=macenko_normalizer,
            target_stats=target_stats,
            patch_size=patch_size,
            read_size=read_size,
            blank_threshold=blank_threshold,
            batch_size=batch_size
        )
        
        if result is not None:
            # 保存h5文件
            h5_path = os.path.join(output_folder, f"{result['slide_name']}.h5")
            save_to_h5(result, h5_path)
            return True, slide_name, None
        else:
            return False, slide_name, "No valid patches found"
            
    except Exception as e:
        return False, slide_name, str(e)

# ======================== 主函数 ========================
def main():
    parser = argparse.ArgumentParser(
        description="Integrated WSI processing: Patch extraction -> Normalization -> Feature extraction",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 输入输出路径
    parser.add_argument('-input_folder', required=True, help='Folder containing WSI files (.ndpi, .svs, or .czi)')
    parser.add_argument('-mask_folder', required=True, help='Folder containing binary masks')
    parser.add_argument('-output_folder', required=True, help='Output folder for h5 feature files')
    
    # Patch提取参数
    parser.add_argument('-patch_size', type=int, default=224,
                       help='Final patch size fed into CONCH after resizing from read_size (recommended: 224)')
    parser.add_argument('-read_size', type=int, default=None, 
                       help='Read size from WSI before resizing (default: 512 for 40x, patch_size for others)')
    parser.add_argument('-blank_threshold', type=float, default=0.9, 
                       help='Blank patch filtering threshold (default: 0.9, same as cut_patch.py)')
    parser.add_argument('--min_mask_ratio', type=float, default=0.0,
                       help='Minimum fraction of positive mask pixels required for a patch candidate. '\
                            'Use 0.1-0.3 to remove edge patches with mostly gray background.')
    parser.add_argument('-file_type', choices=['ndpi', 'svs', 'czi'], default='svs', 
                       help='WSI file type (default: svs)')
    
    # 颜色标准化参数
    parser.add_argument('-r', '--ref_img_path', required=True, 
                       help='Reference image path for color normalization')
    parser.add_argument('-m', '--method', choices=['none', 'reinhard', 'macenko'], default='macenko',
                       help='Normalization method (default: macenko; use none to disable color normalization)')
    
    # 特征提取参数
    parser.add_argument('--checkpoint_path', required=True,
                       help='Path to the CONCH pytorch_model.bin checkpoint')
    parser.add_argument('--batch_size', type=int, default=64, 
                       help='Batch size for feature extraction (default: 64)')
    
    # 性能优化参数
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                       help='Save checkpoint every N batches (default: 50)')
    parser.add_argument('--save_filter_log', action='store_true',
                       help='Save detailed log of filtered patches')
    parser.add_argument('--grid_output_dir', type=str, default=None,
                       help='Directory to save grid visualizations (e.g., /path/to/grid)')
    parser.add_argument('--num_workers', type=int, default=6,
                       help='Number of worker processes for parallel patch loading (default: 6). '\
                            'Note: Macenko normalization is CPU-intensive. More workers = faster but higher CPU usage. '\
                            'Recommended: 6-10 for balanced performance')
    parser.add_argument('--czi_strip_patches', type=int, default=16,
                       help='For CZI only: read up to N contiguous patches in the same row with one ROI (default: 16). '\
                            'Set to 1 to use one ROI per patch.')
    parser.add_argument('--max_slides', type=int, default=None,
                       help='Process at most N slides for debugging/smoke tests. Default: process all slides.')
    
    # Patch保存参数（新增）
    parser.add_argument('--save_raw_patches', type=str, default=None,
                       help='Directory to save raw extracted patches (before normalization). '\
                            'If not specified, raw patches will not be saved.')
    parser.add_argument('--save_normalized_patches', type=str, default=None,
                       help='Directory to save normalized patches (after color normalization). '\
                            'If not specified, normalized patches will not be saved.')
    parser.add_argument('--max_saved_patches', type=int, default=None,
                       help='Save at most N raw/normalized patch images per slide. '\
                            'This only limits debug PNG saving, not feature extraction.')
    
    args = parser.parse_args()
    
    # 设置设备
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"🖥️  Using device: GPU (cuda) - {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print(f"🖥️  Using device: CPU")
    
    # 加载模型
    print(f"\n📦 Loading CONCH model from {args.checkpoint_path}...")
    model = get_conch_model(args.checkpoint_path)
    model = model.to(device)
    print("✅ Model loaded successfully")
    
    # 准备图像转换（不做 resize：patch 会在 read_size 后先 resize 到 patch_size）
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CONCH_MEAN, std=CONCH_STD),
    ])
    
    # 准备标准化器
    print(f"\n🎨 Preparing {args.method} normalizer...")
    macenko_normalizer, target_stats = prepare_normalizer(args.method, args.ref_img_path, device, args.patch_size)
    print("✅ Normalizer ready")
    
    # 获取所有WSI文件
    ext = f".{args.file_type}"
    wsi_files = sorted([f for f in os.listdir(args.input_folder) if f.endswith(ext)])
    
    if len(wsi_files) == 0:
        print(f"❌ No {ext} files found in {args.input_folder}")
        return
    
    print(f"\n📂 Found {len(wsi_files)} {args.file_type.upper()} files")
    if args.max_slides is not None:
        print(f"📌 Debug mode: will process at most {args.max_slides} slide(s) with matching masks")
    
    # 创建输出目录
    os.makedirs(args.output_folder, exist_ok=True)
    
    # 处理每个切片
    print(f"\n🚀 Starting processing...")
    success_count = 0
    failed_slides = []
    attempted_slides = 0

    for idx, wsi_file in enumerate(wsi_files, 1):
        if args.max_slides is not None and attempted_slides >= args.max_slides:
            print(f"\n📌 Reached --max_slides {args.max_slides}; stopping early.")
            break

        slide_name_base = os.path.splitext(wsi_file)[0]
        mask_file_name = f"{slide_name_base}_binary_mask_800x800.png"
        
        slide_path = os.path.join(args.input_folder, wsi_file)
        mask_path = os.path.join(args.mask_folder, mask_file_name)
        
        if not os.path.exists(mask_path):
            print(f"\n[{idx}/{len(wsi_files)}] ⚠️  Mask not found: {mask_file_name}, skipping.")
            failed_slides.append(slide_name_base)
            continue

        attempted_slides += 1
        print(f"\n[{idx}/{len(wsi_files)}] Processing: {wsi_file}")
        
        # 处理切片（带增量保存）
        result = process_single_slide(
            slide_path=slide_path,
            mask_path=mask_path,
            model=model,
            transform=transform,
            device=device,
            method=args.method,
            macenko_normalizer=macenko_normalizer,
            target_stats=target_stats,
            patch_size=args.patch_size,
            read_size=args.read_size,
            blank_threshold=args.blank_threshold,
            batch_size=args.batch_size,
            output_folder=args.output_folder,
            checkpoint_interval=args.checkpoint_interval,
            save_filter_log=args.save_filter_log,
            grid_output_dir=args.grid_output_dir,
            num_workers=args.num_workers,
            ref_img_path=args.ref_img_path,
            save_raw_patches=args.save_raw_patches,
            save_normalized_patches=args.save_normalized_patches,
            file_type=args.file_type,
            czi_strip_patches=args.czi_strip_patches,
            max_saved_patches=args.max_saved_patches,
            min_mask_ratio=args.min_mask_ratio
        )
        
        if result is not None:
            if result.get('skipped'):
                # 已经处理过的slide
                success_count += 1
            else:
                # 新处理的slide（h5已在函数内保存）
                success_count += 1
        else:
            print(f"❌ Failed to process: {wsi_file}")
            failed_slides.append(slide_name_base)
    
    # 最终总结
    print("\n" + "="*60)
    print("🎉 PROCESSING COMPLETED!")
    print("="*60)
    print(f"✅ Successfully processed: {success_count}/{len(wsi_files)} slides")
    if failed_slides:
        print(f"❌ Failed slides: {', '.join(failed_slides)}")
    print(f"📁 Output directory: {args.output_folder}")

if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    main()
