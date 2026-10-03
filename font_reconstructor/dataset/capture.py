"""
Makes clean renderings look like photographs of printed text that were cleaned up again.

A text image that reaches the model in practice has a history: it was printed or shown on a screen, photographed
with a camera, and then preprocessed to a tight crop of white text on black. Each step leaves traces. Strokes
get thicker or thinner, edges get rough, the crop is a bit loose or cuts into the text, a slight rotation is left.

`CaptureSimulation` replays that history on a clean rendering: it distorts and blurs the text, turns it into a
dark-on-light "photo" with uneven lighting, sensor noise and compression, and then runs the same kind of cleanup
a real pipeline runs: background normalization, optional binarization, cropping to the text and resizing.
"""
import io
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

_BLACK_WHITE_LUTS = {}


def ink_bbox(image: Image.Image, threshold: int = 64) -> Optional[Tuple[int, int, int, int]]:
    """
    Bounding box (left, top, right, bottom) of the text of a white-on-black image, None if there is none.

    The image is smoothed first, so that isolated bright pixels of noise do not count as text.
    """
    if threshold not in _BLACK_WHITE_LUTS:
        _BLACK_WHITE_LUTS[threshold] = [255 if value >= threshold else 0 for value in range(256)]
    return image.filter(ImageFilter.BoxBlur(1)).point(_BLACK_WHITE_LUTS[threshold]).getbbox()


def fit_to_box(
    image: Image.Image,
    dims: Tuple[int, int],
    margins: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    align: Tuple[float, float] = (0.5, 0.5),
) -> Image.Image:
    """
    Crop a white-on-black image to its text and scale the crop to fit `dims`, keeping its aspect ratio.

    :param dims: (width, height) of the result
    :param margins: (left, top, right, bottom) background to keep around the text, as fractions of the text
        height. Negative values cut into the text.
    :param align: where the crop is placed if it does not fill the box, (0, 0) is the top left, (1, 1) the
        bottom right
    """
    width, height = dims
    bbox = ink_bbox(image)
    if bbox is None:
        return Image.new("L", (width, height), 0)

    left, top, right, bottom = bbox
    text_height = bottom - top
    left -= round(margins[0] * text_height)
    top -= round(margins[1] * text_height)
    right += round(margins[2] * text_height)
    bottom += round(margins[3] * text_height)
    if right - left < 1 or bottom - top < 1:
        return Image.new("L", (width, height), 0)

    # cropping beyond the image pads with black
    crop = image.crop((left, top, right, bottom))
    scale = min(width / crop.width, height / crop.height)
    size = (max(1, min(width, round(crop.width * scale))), max(1, min(height, round(crop.height * scale))))
    crop = crop.resize(size, Image.LANCZOS)

    result = Image.new("L", (width, height), 0)
    result.paste(crop, (round((width - size[0]) * align[0]), round((height - size[1]) * align[1])))
    return result


def clean_text_image(image: np.ndarray, dims: Tuple[int, int]) -> np.ndarray:
    """
    The ideal preprocessing result of a clean rendering: the text cropped tightly, centred, at full contrast.

    :param image: uint8 array of white text on black
    :param dims: (width, height) of the result
    """
    return np.asarray(fit_to_box(Image.fromarray(image), dims))


class CaptureSimulation:
    """
    Random print-photograph-cleanup history of a text image, see the module docstring.

    All ranges are sampled uniformly. Lengths in pixels refer to the rendering it is given, which should be
    larger than the result (`render_scale` 2 or more), so that blur and stroke changes are finer than one pixel
    of the result.

    :param dims: (width, height) of the result
    :param binarize_prob: share of the images whose cleanup ends with a threshold, as many pipelines do.
        The others keep their grey levels.
    :param max_rotation: largest rotation left after the cleanup, in degrees
    :param max_perspective: largest shift of an image corner, as a fraction of the image size
    :param blur: range of the standard deviation of the optical blur, in pixels
    :param motion_blur_prob: share of the images with motion blur from a moving camera
    :param stroke_change_prob: share of the images whose strokes are thickened or thinned by a pixel
    :param lighting: largest relative change of the brightness from one side of the image to the other
    :param noise: range of the standard deviation of the sensor noise, for a brightness in [0, 1]
    :param min_resolution: lowest resolution of the photo, relative to the rendering
    :param jpeg_prob: share of the images that are compressed
    :param jpeg_quality: range of the compression quality
    :param margin: range of the margin around the text on each side, as a fraction of the text height.
        Negative values cut into the text.
    """

    def __init__(
        self,
        dims: Tuple[int, int],
        binarize_prob: float = 0.5,
        max_rotation: float = 3.0,
        max_perspective: float = 0.03,
        blur: Tuple[float, float] = (0.3, 1.6),
        motion_blur_prob: float = 0.2,
        stroke_change_prob: float = 0.3,
        lighting: float = 0.35,
        noise: Tuple[float, float] = (0.0, 0.05),
        min_resolution: float = 0.5,
        jpeg_prob: float = 0.5,
        jpeg_quality: Tuple[int, int] = (25, 90),
        margin: Tuple[float, float] = (-0.06, 0.25),
    ):
        self.dims = tuple(dims)
        self.binarize_prob = binarize_prob
        self.max_rotation = max_rotation
        self.max_perspective = max_perspective
        self.blur = tuple(blur)
        self.motion_blur_prob = motion_blur_prob
        self.stroke_change_prob = stroke_change_prob
        self.lighting = lighting
        self.noise = tuple(noise)
        self.min_resolution = min_resolution
        self.jpeg_prob = jpeg_prob
        self.jpeg_quality = tuple(jpeg_quality)
        self.margin = tuple(margin)

    def __call__(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        :param image: uint8 array of clean white text on black
        :param rng: the only source of randomness
        :return: uint8 array shaped (height, width) of `dims`, white text on black
        """
        text = Image.fromarray(image)
        text = self._distort(text, rng)
        text = self._print_and_focus(text, rng)
        photo = self._photograph(text, rng)
        cleaned = self._clean_up(photo, rng)

        margins = tuple(rng.uniform(*self.margin, size=4))
        align = (rng.uniform(0.0, 1.0), rng.uniform(0.3, 0.7))
        return np.asarray(fit_to_box(cleaned, self.dims, margins, align))

    def _distort(self, text: Image.Image, rng) -> Image.Image:
        """
        the rotation and perspective that a deskewing step did not remove
        """
        width, height = text.size
        angle = np.deg2rad(rng.uniform(-self.max_rotation, self.max_rotation))
        centre = np.array([width / 2, height / 2])
        corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        moved = (corners - centre) @ rotation.T + centre
        moved += rng.uniform(-self.max_perspective, self.max_perspective, size=(4, 2)) * [width, height]
        return text.transform(
            text.size, Image.PERSPECTIVE, _perspective_coefficients(moved, corners), Image.BILINEAR)

    def _print_and_focus(self, text: Image.Image, rng) -> Image.Image:
        """
        ink spread or loss, lens blur and camera motion
        """
        if rng.random() < self.stroke_change_prob:
            text = text.filter(ImageFilter.MaxFilter(3) if rng.random() < 0.5 else ImageFilter.MinFilter(3))
        text = text.filter(ImageFilter.GaussianBlur(rng.uniform(*self.blur)))
        if rng.random() < self.motion_blur_prob:
            text = text.filter(_motion_kernel(rng))
        return text

    def _photograph(self, text: Image.Image, rng) -> Image.Image:
        """
        dark ink on light paper under uneven light, with sensor noise, limited resolution and compression
        """
        ink = np.asarray(text, dtype=np.float32) / 255.0
        height, width = ink.shape

        paper_level = rng.uniform(0.55, 1.0)
        ink_level = paper_level * rng.uniform(0.05, 0.5)
        scene = paper_level - (paper_level - ink_level) * ink

        x = np.linspace(-0.5, 0.5, width, dtype=np.float32)[None, :]
        y = np.linspace(-0.5, 0.5, height, dtype=np.float32)[:, None]
        slope_x, slope_y = rng.uniform(-self.lighting, self.lighting, size=2)
        scene = scene * (1.0 + slope_x * x + slope_y * y)

        scene = scene + rng.normal(0.0, rng.uniform(*self.noise), size=scene.shape).astype(np.float32)
        photo = Image.fromarray((np.clip(scene, 0.0, 1.0) * 255).astype(np.uint8))

        resolution = rng.uniform(self.min_resolution, 1.0)
        if resolution < 1.0:
            photo = photo.resize((max(8, round(width * resolution)), max(8, round(height * resolution))),
                                 Image.BILINEAR)

        if rng.random() < self.jpeg_prob:
            buffer = io.BytesIO()
            photo.save(buffer, format="JPEG", quality=int(rng.integers(self.jpeg_quality[0], self.jpeg_quality[1] + 1)))
            photo = Image.open(io.BytesIO(buffer.getvalue()))
            photo.load()
        return photo

    def _clean_up(self, photo: Image.Image, rng) -> Image.Image:
        """
        what preprocessing does to a photo: remove the background, invert, stretch the contrast and often binarize
        """
        # the background is the photo without its dark strokes, estimated at a low resolution
        small = photo.resize((max(2, photo.width // 4), max(2, photo.height // 4)), Image.BOX)
        background = small.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.GaussianBlur(2))
        background = np.asarray(background.resize(photo.size, Image.BILINEAR), dtype=np.float32)

        ink = (background - np.asarray(photo, dtype=np.float32)) / np.maximum(background, 1.0)

        # stretch the contrast from just above the noise of the background to the darkest strokes. The peak is
        # read from a smoothed image, so that single noisy pixels do not set it.
        noise_level = 1.4826 * float(np.median(np.abs(ink)))
        floor = rng.uniform(1.0, 3.0) * noise_level
        smoothed = Image.fromarray((np.clip(ink, 0.0, 1.0) * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(1))
        peak = max(float(np.asarray(smoothed).max()) / 255.0, floor + 0.05)
        ink = np.clip((ink - floor) / (peak - floor), 0.0, 1.0)

        if rng.random() < self.binarize_prob:
            # a threshold that is off moves every edge, which makes all strokes thicker or thinner
            threshold = np.clip(_otsu_threshold(ink) * rng.uniform(0.7, 1.3), 0.15, 0.85)
            ink = (ink > threshold).astype(np.float32)
        else:
            ink = ink ** rng.uniform(0.6, 1.7)

        return Image.fromarray((ink * 255).astype(np.uint8))


def _perspective_coefficients(output_points, input_points):
    """
    coefficients for PIL's perspective transform, which maps each output point to the input point it shows
    """
    rows, values = [], []
    for (x, y), (u, v) in zip(output_points, input_points):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        values.extend([u, v])
    return np.linalg.solve(np.array(rows, dtype=np.float64), np.array(values, dtype=np.float64)).tolist()


def _motion_kernel(rng) -> ImageFilter.Kernel:
    """
    a 5x5 kernel that smears the image along a line through its centre
    """
    kernel = np.zeros((5, 5), dtype=np.float64)
    length = int(rng.integers(1, 3))  # pixels to each side of the centre
    direction = [(0, 1), (1, 0), (1, 1), (1, -1)][int(rng.integers(0, 4))]
    for step in range(-length, length + 1):
        kernel[2 + step * direction[0], 2 + step * direction[1]] = 1.0
    return ImageFilter.Kernel((5, 5), kernel.flatten().tolist(), scale=float(kernel.sum()))


def _otsu_threshold(values: np.ndarray, bins: int = 64) -> float:
    """
    the threshold between background and ink that separates their brightness best
    """
    histogram, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    histogram = histogram.astype(np.float64)
    centres = (edges[:-1] + edges[1:]) / 2

    weight_low = np.cumsum(histogram)
    weight_high = weight_low[-1] - weight_low
    sum_low = np.cumsum(histogram * centres)
    mean_low = sum_low / np.maximum(weight_low, 1e-9)
    mean_high = (sum_low[-1] - sum_low) / np.maximum(weight_high, 1e-9)

    between_class_variance = weight_low * weight_high * (mean_low - mean_high) ** 2
    return float(centres[int(np.argmax(between_class_variance))])
