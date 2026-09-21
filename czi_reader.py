from pathlib import Path

import numpy as np
from PIL import Image


class CziSlideReader:
    """Expose the small OpenSlide-like API used by the web pipeline."""

    def __init__(self, slide_path: str | Path):
        try:
            from pylibCZIrw import czi
        except ImportError as exc:
            raise ImportError(
                "Reading .czi files requires pylibCZIrw. "
                "Install it with: pip install pylibCZIrw"
            ) from exc

        self._context = czi.open_czi(str(slide_path))
        self._reader = self._context.__enter__()
        bbox = self._reader.total_bounding_box_no_pyramid
        self.x0 = int(bbox["X"][0])
        self.y0 = int(bbox["Y"][0])
        self.dimensions = (
            int(bbox["X"][1] - bbox["X"][0]),
            int(bbox["Y"][1] - bbox["Y"][0]),
        )
        self.level_dimensions = [self.dimensions]
        self.level_downsamples = [1.0]
        self.properties = {}

    @staticmethod
    def _to_rgb(array) -> Image.Image:
        image = np.asarray(array)
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.shape[-1] > 3:
            image = image[..., :3]
        # pylibCZIrw returns BGR for these slides.
        return Image.fromarray(image[..., ::-1].copy()).convert("RGB")

    def read_region(self, location, level, size) -> Image.Image:
        if level != 0:
            raise ValueError("CZI patch extraction supports level 0 coordinates only")
        x, y = location
        width, height = size
        image = self._reader.read(
            roi=(self.x0 + int(x), self.y0 + int(y), int(width), int(height)),
            zoom=1.0,
            background_pixel=(1.0, 1.0, 1.0),
        )
        return self._to_rgb(image)

    def get_thumbnail(self, size) -> Image.Image:
        max_width, max_height = size
        zoom = min(
            max_width / self.dimensions[0],
            max_height / self.dimensions[1],
            1.0,
        )
        image = self._reader.read(
            zoom=zoom,
            background_pixel=(1.0, 1.0, 1.0),
        )
        return self._to_rgb(image)

    def close(self):
        if self._context is not None:
            self._context.__exit__(None, None, None)
            self._context = None


def open_slide(slide_path: str | Path, file_type: str | None = None):
    path = Path(slide_path)
    if (file_type or path.suffix.lstrip(".")).lower() == "czi":
        return CziSlideReader(path)

    import openslide

    return openslide.OpenSlide(str(path))
