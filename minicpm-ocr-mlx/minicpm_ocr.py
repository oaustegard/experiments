"""MiniCPM-OCR 1.0 for mlx-vlm.

MiniCPM-OCR 1.0 is the MiniCPM-V 4.6 architecture (SigLIP vision tower, 2x2
merger, Qwen3.5 hybrid text model) run in its "4x" downsample mode, with an
image processor that sizes images for that mode. transformers declares it as
plain subclasses of the 4.6 classes, so this module reuses mlx-vlm's
`minicpmv4_6` port and fixes the parts that assume "16x":

- the 4.6 image processor always rounds sizes to 56 px (`patch_size * 4`, the
  16x merge); 4x needs 28 px, otherwise slices get the wrong token counts;
- its slice-grid search breaks ties differently from transformers.

A converted checkpoint names this file in `config.json` as `model_file`, and
mlx-vlm imports it in place of a built-in model package. Importing it also
registers the processor and the prompt format for `model_type: minicpm_ocr`.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from mlx_vlm import prompt_utils
from mlx_vlm.models.base import install_auto_processor_patch
from mlx_vlm.models.minicpmv4_6 import (
    LanguageModel,
    MiniCPMVImageProcessor,
    MiniCPMVProcessor,
    MiniCPMVVideoProcessor,
    Model,
    SliceConfig,
    TextConfig,
    VisionModel,
)
from mlx_vlm.models.minicpmv4_6 import ModelConfig as _ModelConfigV46
from mlx_vlm.models.minicpmv4_6 import VisionConfig as _VisionConfigV46


@dataclass
class VisionConfig(_VisionConfigV46):
    def __post_init__(self):
        if self.model_type == "minicpm_ocr_vision":
            self.model_type = "minicpmv4_6_vision"
        super().__post_init__()


@dataclass
class ModelConfig(_ModelConfigV46):
    model_type: str = "minicpm_ocr"
    downsample_mode: str = "4x"


class MiniCPMOcrImageProcessor(MiniCPMVImageProcessor):
    """Sizes images as transformers' `MiniCPMOcrImageProcessor` does.

    Supports `small_image_upsize` "always" (the default) and "never". The
    opt-in "adaptive" mode needs OpenCV glyph statistics and per-slice sizes
    and is not ported.
    """

    def __init__(self, downsample_mode: str = "4x", small_image_upsize: str = "always", **kwargs):
        if small_image_upsize not in ("always", "never"):
            raise NotImplementedError(f"small_image_upsize={small_image_upsize!r} is not ported to MLX")
        super().__init__(downsample_mode=downsample_mode, **kwargs)
        self.small_image_upsize = small_image_upsize

    def _find_best_resize(
        self,
        image_size: Tuple[int, int],
        scale_resolution: int,
        patch_size: int,
        allow_upscale: bool = False,
    ) -> Tuple[int, int]:
        width, height = image_size
        if width * height > scale_resolution * scale_resolution or allow_upscale:
            aspect_ratio = width / height
            height = int(scale_resolution / math.sqrt(aspect_ratio))
            width = int(height * aspect_ratio)
        # "4x" merges one 2x2 block, "16x" merges twice (ViT merger + downsample MLP)
        unit = patch_size * (2 if self.downsample_mode == "4x" else 4)
        return self._ensure_divide(width, unit), self._ensure_divide(height, unit)

    def get_sliced_grid(
        self,
        image_size: Tuple[int, int],
        max_slice_nums: Optional[int] = None,
        never_split: bool = False,
    ) -> Optional[Tuple[int, int]]:
        width, height = image_size
        max_slice_nums = self.max_slice_nums if max_slice_nums is None else int(max_slice_nums)
        ratio = width * height / (self.scale_resolution * self.scale_resolution)
        multiple = min(math.ceil(ratio), max_slice_nums)
        if multiple <= 1 or never_split:
            return None

        # transformers' search and tie-break, which returns (rows, cols)
        log_ratio = math.log(width / height)
        best_rows, best_cols = 1, 1
        min_error = float("inf")
        for num_slices in (multiple - 1, multiple, multiple + 1):
            if num_slices == 1 or num_slices > max_slice_nums:
                continue
            for rows in range(1, num_slices + 1):
                if num_slices % rows:
                    continue
                cols = num_slices // rows
                error = abs(log_ratio - math.log(cols / rows))
                if error < min_error or (error == min_error and rows > best_rows):
                    best_rows, best_cols = rows, cols
                    min_error = error
        return best_cols, best_rows

    def slice_image(self, image, max_slice_nums=None, never_split=False):
        source, patches, grid = super().slice_image(image, max_slice_nums, never_split)
        if grid is None and self.small_image_upsize == "never":
            size = self._find_best_resize(image.size, self.scale_resolution, self.patch_size, allow_upscale=False)
            source = image.resize(size, resample=self._resample)
        return source, patches, grid

    _resample = 3  # PIL.Image.Resampling.BICUBIC


class MiniCPMOcrProcessor(MiniCPMVProcessor):
    @staticmethod
    def _strip_think_prefix(text):
        # The 4.6 port drops the template's empty <think> block; transformers and
        # vLLM keep it, and the OCR checkpoint decodes cleanly with it.
        return text

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        processor = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        processor.image_processor = MiniCPMOcrImageProcessor.from_pretrained(pretrained_model_name_or_path)
        return processor


install_auto_processor_patch(["minicpm_ocr"], MiniCPMOcrProcessor)
# The 4.6 entry wraps images as "(<image>./</image>)", which leaves the parentheses
# in the prompt as tokens; a content list renders through the checkpoint's own
# chat template, as transformers does.
prompt_utils.MODEL_CONFIG.setdefault("minicpm_ocr", prompt_utils.MessageFormat.LIST_WITH_IMAGE_FIRST)

__all__ = [
    "LanguageModel",
    "MiniCPMOcrImageProcessor",
    "MiniCPMOcrProcessor",
    "MiniCPMVVideoProcessor",
    "Model",
    "ModelConfig",
    "SliceConfig",
    "TextConfig",
    "VisionConfig",
    "VisionModel",
]
