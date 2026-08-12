"""Deterministic extraction of numeric series from raster line charts.

The visual tracer is intentionally independent from OCR.  Callers provide a
small recognizer adapter which receives tightly cropped tick labels.  Marker's
PDF processor uses the existing Surya recognition model, while the standalone
CLI can also use a locally installed Tesseract binary.
"""

from __future__ import annotations

import colorsys
import html
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import pairwise
from statistics import median
from typing import Any, Protocol

from PIL import Image, ImageDraw, ImageOps


class ChartError(RuntimeError):
    """A recoverable chart detection, OCR, or calibration error."""


@dataclass(frozen=True)
class PlotBox:
    """Inclusive pixel bounds of a chart's plot area."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def validate(self, image_width: int, image_height: int) -> None:
        if self.width < 10 or self.height < 10:
            raise ChartError("plot bounds must span at least 10 pixels per axis")
        if not (
            0 <= self.left < self.right < image_width
            and 0 <= self.top < self.bottom < image_height
        ):
            raise ChartError(
                f"plot bounds {self} fall outside {image_width}x{image_height} image"
            )


@dataclass(frozen=True)
class Axis:
    """Manual axis bounds used when OCR calibration is not requested."""

    minimum: float
    maximum: float
    scale: str = "linear"

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ChartError("axis bounds must be finite")
        if self.minimum >= self.maximum:
            raise ChartError("axis minimum must be smaller than maximum")
        if self.scale not in {"linear", "log10"}:
            raise ChartError(f"unsupported axis scale: {self.scale}")
        if self.scale == "log10" and self.minimum <= 0:
            raise ChartError("log10 axes require positive bounds")

    def fraction_to_value(self, fraction: float) -> float:
        fraction = min(1.0, max(0.0, fraction))
        if self.scale == "linear":
            return self.minimum + fraction * (self.maximum - self.minimum)
        low = math.log10(self.minimum)
        high = math.log10(self.maximum)
        return 10 ** (low + fraction * (high - low))

    def value_to_fraction(self, value: float) -> float:
        if self.scale == "linear":
            return (value - self.minimum) / (self.maximum - self.minimum)
        if value <= 0:
            raise ChartError("log10 axis values must be positive")
        low = math.log10(self.minimum)
        high = math.log10(self.maximum)
        return (math.log10(value) - low) / (high - low)


@dataclass(frozen=True)
class Calibration:
    """Calibration from explicit axis bounds."""

    x: Axis
    y: Axis

    def x_value(self, pixel_x: float, plot: PlotBox) -> float:
        return self.x.fraction_to_value((pixel_x - plot.left) / plot.width)

    def y_value(self, pixel_y: float, plot: PlotBox) -> float:
        return self.y.fraction_to_value((plot.bottom - pixel_y) / plot.height)

    def y_pixel(self, value: float, plot: PlotBox) -> float:
        return plot.bottom - self.y.value_to_fraction(value) * plot.height


@dataclass(frozen=True)
class Tick:
    axis: str
    pixel: float
    value: float
    text: str
    confidence: float | None = None
    inlier: bool = True


@dataclass(frozen=True)
class FittedAxis:
    """Affine pixel transform fitted in linear or log10 value space."""

    scale: str
    slope: float
    intercept: float
    normalized_rmse: float
    ticks: tuple[Tick, ...]

    def pixel_to_value(self, pixel: float) -> float:
        transformed = self.slope * pixel + self.intercept
        return transformed if self.scale == "linear" else 10**transformed

    def value_to_pixel(self, value: float) -> float:
        if self.scale == "log10":
            if value <= 0:
                raise ChartError("log10 axis values must be positive")
            value = math.log10(value)
        return (value - self.intercept) / self.slope


@dataclass(frozen=True)
class TickCalibration:
    x: FittedAxis
    y: FittedAxis

    def x_value(self, pixel_x: float, plot: PlotBox) -> float:
        return self.x.pixel_to_value(pixel_x)

    def y_value(self, pixel_y: float, plot: PlotBox) -> float:
        return self.y.pixel_to_value(pixel_y)

    def y_pixel(self, value: float, plot: PlotBox) -> float:
        return self.y.value_to_pixel(value)


class CalibrationLike(Protocol):
    def x_value(self, pixel_x: float, plot: PlotBox) -> float: ...

    def y_value(self, pixel_y: float, plot: PlotBox) -> float: ...

    def y_pixel(self, value: float, plot: PlotBox) -> float: ...


@dataclass(frozen=True)
class SeriesSpec:
    name: str
    color: tuple[int, int, int] | None = None
    color_tolerance: float = 90.0
    minimum_chroma: int = 12


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    pixel_x: int
    pixel_y: float
    confidence: float
    observed: bool


@dataclass(frozen=True)
class Trace:
    name: str
    requested_color: str | None
    points: tuple[Point, ...]
    observed_columns: int
    plot_columns: int

    @property
    def coverage(self) -> float:
        return self.observed_columns / self.plot_columns if self.plot_columns else 0.0


@dataclass(frozen=True)
class TickCrop:
    axis: str
    pixel: float
    bbox: tuple[int, int, int, int]
    image: Image.Image


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float | None = None


class TickRecognizer(Protocol):
    def recognize(self, crops: Sequence[TickCrop]) -> Sequence[OCRResult]: ...


def parse_color(value: str) -> tuple[int, int, int]:
    normalized = value.strip().removeprefix("#")
    if len(normalized) != 6:
        raise ChartError(f"color must be #RRGGBB, got {value!r}")
    try:
        return tuple(int(normalized[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError as exc:
        raise ChartError(f"color must be #RRGGBB, got {value!r}") from exc


def format_color(color: tuple[int, int, int] | None) -> str | None:
    if color is None:
        return None
    return "#" + "".join(f"{component:02x}" for component in color)


def _group_adjacent(values: Iterable[int], maximum_gap: int = 1) -> list[list[int]]:
    groups: list[list[int]] = []
    for value in values:
        if not groups or value > groups[-1][-1] + maximum_gap:
            groups.append([value])
        else:
            groups[-1].append(value)
    return groups


def _neutral_pixel(pixel: tuple[int, int, int], maximum: int = 253) -> bool:
    return max(pixel) - min(pixel) <= 12 and 20 <= max(pixel) <= maximum


def detect_plot_box(image: Image.Image) -> PlotBox:
    """Detect a rectangular plot frame from long neutral border lines."""

    rgb = image.convert("RGB")
    width, height = rgb.size
    pixels = rgb.load()
    row_scores = [
        sum(_neutral_pixel(pixels[x, y], 249) for x in range(width))
        for y in range(height)
    ]
    row_candidates: list[int] = []
    for ratio in (0.55, 0.45, 0.35):
        row_candidates = [
            y for y, score in enumerate(row_scores) if score >= width * ratio
        ]
        if len(_group_adjacent(row_candidates)) >= 2:
            break
    row_groups = _group_adjacent(row_candidates)
    if len(row_groups) < 2:
        raise ChartError("could not detect top and bottom plot borders")
    horizontal_lines = [max(group, key=row_scores.__getitem__) for group in row_groups]
    top = min(horizontal_lines)
    bottom = max(horizontal_lines)
    if bottom - top < height * 0.35:
        raise ChartError("detected horizontal plot borders are implausibly close")

    span = bottom - top + 1
    column_scores = [
        sum(_neutral_pixel(pixels[x, y], 249) for y in range(top, bottom + 1))
        for x in range(width)
    ]
    column_candidates: list[int] = []
    for ratio in (0.72, 0.60, 0.48):
        column_candidates = [
            x for x, score in enumerate(column_scores) if score >= span * ratio
        ]
        if column_candidates:
            break
    if not column_candidates:
        raise ChartError("could not detect vertical plot borders")
    column_groups = _group_adjacent(column_candidates)
    vertical_lines = [
        max(group, key=column_scores.__getitem__) for group in column_groups
    ]
    left = min(vertical_lines)
    right = max(vertical_lines)
    if right < width * 0.80:
        right = width - 1
    if left > width * 0.20:
        left = 0

    plot = PlotBox(left=left, top=top, right=right, bottom=bottom)
    plot.validate(width, height)
    return plot


def _grid_line_positions(image: Image.Image, plot: PlotBox, axis: str) -> list[int]:
    """Find major grid lines; unlabeled borders are filtered after OCR."""

    rgb = image.convert("RGB")
    pixels = rgb.load()
    if axis == "y":
        positions = range(plot.top, plot.bottom + 1)
        length = plot.width + 1

        def score(position: int, maximum: int) -> int:
            return sum(
                _neutral_pixel(pixels[x, position], maximum)
                for x in range(plot.left, plot.right + 1)
            )

        maxima = (253, 254, 251, 249)
        minimum_lines = 3
    else:
        positions = range(plot.left, plot.right + 1)
        length = plot.height + 1

        def score(position: int, maximum: int) -> int:
            return sum(
                _neutral_pixel(pixels[position, y], maximum)
                for y in range(plot.top, plot.bottom + 1)
            )

        # The major decade lines in datasheet plots are normally darker than
        # logarithmic minor grid lines.  Starting at 251 keeps the OCR crops
        # wide enough to contain an entire centered label.
        maxima = (251, 250, 252, 253)
        minimum_lines = 2

    selected: list[int] = []
    for maximum in maxima:
        scores = {position: score(position, maximum) for position in positions}
        candidates = [
            position for position, value in scores.items() if value >= length * 0.72
        ]
        groups = _group_adjacent(candidates, maximum_gap=5 if axis == "y" else 2)
        selected = [max(group, key=scores.__getitem__) for group in groups]
        if len(selected) >= minimum_lines:
            break
    if len(selected) < minimum_lines:
        raise ChartError(f"could not detect enough {axis}-axis grid lines")

    # Always keep the detected frame even when JPEG ringing moves the local
    # maximum by a pixel.
    bounds = (plot.top, plot.bottom) if axis == "y" else (plot.left, plot.right)
    for bound in bounds:
        if all(abs(bound - position) > 2 for position in selected):
            selected.append(bound)
    selected.sort()
    return selected


def _occupied_groups(
    values: Sequence[int], minimum: int = 1, maximum_gap: int = 3
) -> list[list[int]]:
    return _group_adjacent(
        (index for index, value in enumerate(values) if value >= minimum),
        maximum_gap=maximum_gap,
    )


def _y_label_band(image: Image.Image, plot: PlotBox) -> tuple[int, int]:
    # Tick labels are right-aligned immediately beside the plot.  Keeping this
    # band narrow excludes a rotated y-axis title, which otherwise confuses OCR
    # on the small datasheet figures.
    width = max(18, round(plot.height * 0.08))
    return max(0, plot.left - width), plot.left


def _x_label_band(image: Image.Image, plot: PlotBox) -> tuple[int, int]:
    gray = image.convert("L")
    pixels = gray.load()
    rows = range(plot.bottom + 1, image.height)
    counts = [
        sum(pixels[x, y] < 210 for x in range(plot.left, plot.right + 1)) for y in rows
    ]
    groups = [
        group for group in _occupied_groups(counts, minimum=2, maximum_gap=1) if group
    ]
    if not groups:
        return plot.bottom + 1, min(image.height, plot.bottom + 24)
    # JPEG extraction can move one or two rows of a thick bottom frame just
    # outside the detected plot.  Such a row spans nearly the complete plot
    # and must not be mistaken for the first row of tick labels.
    label_groups = [
        group
        for group in groups
        if max(counts[index] for index in group) < (plot.width + 1) * 0.60
    ]
    group = (label_groups or groups)[0]
    return (
        max(plot.bottom + 1, plot.bottom + 1 + group[0] - 2),
        min(image.height, plot.bottom + 1 + group[-1] + 3),
    )


def _x_label_positions(
    image: Image.Image,
    plot: PlotBox,
    band: tuple[int, int],
    grid_lines: Sequence[int],
) -> list[int]:
    """Locate centered x labels and snap their centers to nearby grid lines."""

    gray = image.convert("L")
    pixels = gray.load()
    start = max(0, plot.left - max(20, round(plot.width * 0.08)))
    stop = min(image.width, plot.right + max(20, round(plot.width * 0.08)))
    occupied = [
        x for x in range(start, stop) if any(pixels[x, y] < 210 for y in range(*band))
    ]
    # A JPEG-compressed digit can be split by a four-pixel white gap.  Joining
    # that gap keeps labels such as 1000 and 10000 in one crop.
    groups = _group_adjacent(occupied, maximum_gap=4)
    maximum_width = max(18, (band[1] - band[0]) * 3)
    labels = []
    for group in groups:
        if len(group) < 2 or group[-1] - group[0] + 1 > maximum_width:
            continue
        center = round((group[0] + group[-1]) / 2)
        labels.append((center, group[-1] - group[0] + 1))

    # Plain decimal decades grow by roughly one glyph per label.  Their text
    # center is more trustworthy than a grid line partly obscured by a curve;
    # compact scientific-notation labels benefit from a wider snap tolerance.
    widths = [width for _, width in labels]
    growing_decimal_labels = len(widths) >= 3 and (
        sum(right >= left + 4 for left, right in pairwise(widths))
        >= len(widths) - 2
        and widths[-1] >= widths[0] * 1.60
    )
    snap_tolerance = 3 if growing_decimal_labels else 7
    positions = []
    for center, _ in labels:
        nearby = min(grid_lines, key=lambda line: abs(line - center))
        positions.append(
            nearby if abs(nearby - center) <= snap_tolerance else center
        )
    return sorted(set(positions))


def find_tick_crops(image: Image.Image, plot: PlotBox) -> list[TickCrop]:
    """Create one OCR crop per visible major grid line."""

    x_grid_lines = _grid_line_positions(image, plot, "x")
    y_lines = _grid_line_positions(image, plot, "y")
    y_text_left, y_text_right = _y_label_band(image, plot)
    x_text_top, x_text_bottom = _x_label_band(image, plot)
    x_lines = _x_label_positions(image, plot, (x_text_top, x_text_bottom), x_grid_lines)
    if len(x_lines) < 2:
        x_lines = x_grid_lines
    crops: list[TickCrop] = []

    for index, pixel_y in enumerate(y_lines):
        before = pixel_y - y_lines[index - 1] if index else None
        after = y_lines[index + 1] - pixel_y if index + 1 < len(y_lines) else None
        spacing = min(value for value in (before, after) if value is not None)
        half_height = max(6, min(18, round(spacing * 0.38)))
        bbox = (
            y_text_left,
            max(0, pixel_y - half_height),
            y_text_right,
            min(image.height, pixel_y + half_height + 1),
        )
        crops.append(TickCrop("y", float(pixel_y), bbox, image.crop(bbox)))

    for index, pixel_x in enumerate(x_lines):
        before = pixel_x - x_lines[index - 1] if index else None
        after = x_lines[index + 1] - pixel_x if index + 1 < len(x_lines) else None
        spacing = min(value for value in (before, after) if value is not None)
        half_width = max(12, min(90, round(spacing * 0.44)))
        bbox = (
            max(0, pixel_x - half_width),
            x_text_top,
            min(image.width, pixel_x + half_width + 1),
            x_text_bottom,
        )
        crops.append(TickCrop("x", float(pixel_x), bbox, image.crop(bbox)))
    return crops


_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻", "0123456789+-")
_SUPERSCRIPTS = "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻"


def _ocr_text(value: str) -> str:
    value = html.unescape(value or "")
    value = re.sub(
        r"<sup[^>]*>(.*?)</sup>",
        lambda match: "^(" + re.sub(r"<[^>]+>", "", match.group(1)) + ")",
        value,
        flags=re.IGNORECASE | re.DOTALL,
    )
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.replace("−", "-").replace("–", "-").replace("—", "-")
    value = re.sub(
        rf"([{_SUPERSCRIPTS}]+)",
        lambda match: "^(" + match.group(1).translate(_SUPERSCRIPT_MAP) + ")",
        value,
    )
    return " ".join(value.split())


def parse_tick_value(value: str) -> float | None:
    """Parse plain, scientific, HTML-superscript, and Unicode tick labels."""

    text = _ocr_text(value)
    text = text.replace("\\times", "×").replace("\\cdot", "×")
    text = text.replace("{", "(").replace("}", ")")
    text = re.sub(r"(?<=\d)[ ,](?=\d{3}(?:\D|$))", "", text)
    power = re.search(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(?:×\s*)?10\s*\^\s*\(?\s*([+-]?\d+)\s*\)?",
        text,
        flags=re.IGNORECASE,
    )
    if power:
        try:
            return float(power.group(1)) * 10 ** int(power.group(2))
        except (OverflowError, ValueError):
            return None
    simple_power = re.search(
        r"\b10\s*\^\s*\(?\s*([+-]?\d+)\s*\)?", text, flags=re.IGNORECASE
    )
    if simple_power:
        try:
            return 10 ** int(simple_power.group(1))
        except (OverflowError, ValueError):
            return None
    scientific = re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)[eE][+-]?\d+", text)
    if scientific:
        try:
            return float(scientific.group(0))
        except ValueError:
            return None
    signed_number = re.search(r"(?<!\d)-\s*(?:\d+(?:\.\d*)?|\.\d+)", text)
    if signed_number:
        try:
            return float(signed_number.group(0).replace(" ", ""))
        except ValueError:
            return None
    number = re.search(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text)
    if not number:
        return None
    try:
        return float(number.group(0))
    except ValueError:
        return None


def _least_squares(points: Sequence[tuple[float, float]]) -> tuple[float, float]:
    mean_x = sum(point[0] for point in points) / len(points)
    mean_y = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - mean_x) ** 2 for point in points)
    if denominator <= 1e-12:
        raise ChartError("tick positions do not span enough pixels")
    slope = (
        sum((point[0] - mean_x) * (point[1] - mean_y) for point in points) / denominator
    )
    return slope, mean_y - slope * mean_x


def _fit_candidate(
    ticks: Sequence[Tick], scale: str, direction: str
) -> tuple[float, float, float, set[int]] | None:
    usable = [
        (index, tick)
        for index, tick in enumerate(ticks)
        if scale == "linear" or tick.value > 0
    ]
    if len(usable) < 2:
        return None
    transformed = [
        math.log10(tick.value) if scale == "log10" else tick.value for _, tick in usable
    ]
    nonzero_steps = [
        abs(right - left)
        for left, right in zip(sorted(set(transformed)), sorted(set(transformed))[1:])
        if abs(right - left) > 1e-12
    ]
    if not nonzero_steps:
        return None
    typical_step = median(nonzero_steps)
    tolerance = max(typical_step * 0.18, 1e-8)
    best: tuple[int, float, float, float, set[int]] | None = None

    for left_index, (_, left) in enumerate(usable):
        for right_index in range(left_index + 1, len(usable)):
            _, right = usable[right_index]
            if abs(right.pixel - left.pixel) < 1e-9:
                continue
            left_value = transformed[left_index]
            right_value = transformed[right_index]
            if abs(right_value - left_value) < 1e-12:
                continue
            slope = (right_value - left_value) / (right.pixel - left.pixel)
            if direction == "increasing" and slope <= 0:
                continue
            if direction == "decreasing" and slope >= 0:
                continue
            intercept = left_value - slope * left.pixel
            inliers = {
                index
                for index, ((_, tick), value) in enumerate(zip(usable, transformed))
                if abs(slope * tick.pixel + intercept - value) <= tolerance
            }
            if len(inliers) < 2:
                continue
            squared_error = sum(
                (slope * usable[index][1].pixel + intercept - transformed[index]) ** 2
                for index in inliers
            )
            candidate = (len(inliers), -squared_error, slope, intercept, inliers)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        return None

    inlier_points = [
        (usable[index][1].pixel, transformed[index]) for index in sorted(best[4])
    ]
    slope, intercept = _least_squares(inlier_points)
    errors = [slope * pixel + intercept - value for pixel, value in inlier_points]
    value_range = max(value for _, value in inlier_points) - min(
        value for _, value in inlier_points
    )
    normalized_rmse = (
        math.sqrt(sum(error * error for error in errors) / len(errors)) / value_range
        if value_range > 0
        else math.inf
    )
    original_indices = {usable[index][0] for index in best[4]}
    return slope, intercept, normalized_rmse, original_indices


def fit_axis(
    ticks: Sequence[Tick],
    scale_hint: str = "auto",
    direction: str = "either",
) -> FittedAxis:
    if scale_hint not in {"auto", "linear", "log10"}:
        raise ChartError(f"unsupported axis scale hint: {scale_hint}")
    if direction not in {"increasing", "decreasing", "either"}:
        raise ChartError(f"unsupported axis direction: {direction}")
    candidates = []
    scales = ("linear", "log10") if scale_hint == "auto" else (scale_hint,)
    for scale in scales:
        fit = _fit_candidate(ticks, scale, direction)
        if fit is None:
            continue
        slope, intercept, error, inliers = fit
        # Inlier count dominates.  With only two labels the models are
        # indistinguishable, so auto mode deliberately prefers linear.
        preference = 1 if scale == "linear" else 0
        candidates.append((len(inliers), -error, preference, scale, fit))
    if not candidates:
        raise ChartError("OCR produced fewer than two consistent numeric ticks")
    _, _, _, scale, fit = max(candidates, key=lambda item: item[:3])
    slope, intercept, error, inliers = fit
    annotated = tuple(
        replace(tick, inlier=index in inliers) for index, tick in enumerate(ticks)
    )
    return FittedAxis(
        scale=scale,
        slope=slope,
        intercept=intercept,
        normalized_rmse=error,
        ticks=annotated,
    )


def infer_tick_calibration(
    image: Image.Image,
    recognizer: TickRecognizer,
    *,
    plot: PlotBox | None = None,
    x_scale: str = "auto",
    y_scale: str = "auto",
) -> tuple[PlotBox, TickCalibration]:
    plot = plot or detect_plot_box(image)
    crops = find_tick_crops(image, plot)
    results = list(recognizer.recognize(crops))
    if len(results) != len(crops):
        raise ChartError(
            f"tick recognizer returned {len(results)} results for {len(crops)} crops"
        )
    ticks: dict[str, list[Tick]] = {"x": [], "y": []}
    for crop, result in zip(crops, results):
        value = parse_tick_value(result.text)
        if value is None or not math.isfinite(value):
            continue
        ticks[crop.axis].append(
            Tick(
                axis=crop.axis,
                pixel=crop.pixel,
                value=value,
                text=_ocr_text(result.text),
                confidence=result.confidence,
            )
        )
    return plot, TickCalibration(
        x=fit_axis(ticks["x"], x_scale, direction="increasing"),
        y=fit_axis(ticks["y"], y_scale, direction="decreasing"),
    )


def _pixel_strength(pixel: tuple[int, int, int], spec: SeriesSpec) -> float:
    chroma = max(pixel) - min(pixel)
    if spec.color is None:
        if chroma < spec.minimum_chroma:
            return 0.0
        return min(1.0, chroma / 55.0)
    distance = math.sqrt(
        sum((component - target) ** 2 for component, target in zip(pixel, spec.color))
    )
    if distance >= spec.color_tolerance:
        return 0.0
    target_chroma = max(spec.color) - min(spec.color)
    distance_strength = 1.0 - distance / spec.color_tolerance
    if target_chroma < 18:
        # Explicit neutral colors are intentional series (for example a gray
        # response curve), not missing chroma.  Squaring the distance score
        # keeps black grid lines and their light antialiasing from dominating.
        return distance_strength**2
    if chroma < spec.minimum_chroma:
        return 0.0
    return min(1.0, chroma / 55.0) * distance_strength


def _column_segments(
    pixels: Any, pixel_x: int, plot: PlotBox, spec: SeriesSpec
) -> list[tuple[float, float]]:
    candidates = []
    strengths: dict[int, float] = {}
    for pixel_y in range(plot.top + 1, plot.bottom):
        strength = _pixel_strength(pixels[pixel_x, pixel_y], spec)
        if strength >= 0.12:
            candidates.append(pixel_y)
            strengths[pixel_y] = strength
    if not candidates:
        return []
    groups = _group_adjacent(candidates, maximum_gap=2)
    segments: list[tuple[float, float]] = []
    neutral_target = spec.color is not None and max(spec.color) - min(spec.color) < 18
    for group in groups:
        # A one-pixel neutral fringe is usually antialiasing around a black
        # grid line.  Real gray strokes occupy at least two adjacent rows.
        if neutral_target and (
            len(group) < 2 or len(group) > plot.height * 0.55
        ):
            continue
        weight = sum(strengths[pixel_y] for pixel_y in group)
        center = sum(pixel_y * strengths[pixel_y] for pixel_y in group) / weight
        confidence = min(1.0, max(strengths[pixel_y] for pixel_y in group))
        segments.append((center, confidence))
    return segments


def _interpolate_short_gaps(
    samples: list[tuple[float, float, bool] | None], maximum_gap: int
) -> list[tuple[float, float, bool] | None]:
    result = list(samples)
    index = 0
    while index < len(result):
        if result[index] is not None:
            index += 1
            continue
        start = index
        while index < len(result) and result[index] is None:
            index += 1
        end = index
        if (
            start > 0
            and end < len(result)
            and end - start <= maximum_gap
            and result[start - 1] is not None
            and result[end] is not None
        ):
            left = result[start - 1]
            right = result[end]
            assert left is not None and right is not None
            for offset, gap_index in enumerate(range(start, end), start=1):
                fraction = offset / (end - start + 1)
                center = left[0] + fraction * (right[0] - left[0])
                confidence = min(left[1], right[1]) * 0.5
                result[gap_index] = (center, confidence, False)
    return result


def trace_series(
    image: Image.Image,
    plot: PlotBox,
    calibration: CalibrationLike,
    spec: SeriesSpec,
    *,
    maximum_gap: int = 6,
    smoothness: float = 0.22,
) -> Trace:
    """Trace one chromatically distinct series as a y=f(x) polyline."""

    rgb = image.convert("RGB")
    pixels = rgb.load()
    samples: list[tuple[float, float, bool] | None] = []
    previous_y: float | None = None
    observed_columns = 0
    for pixel_x in range(plot.left, plot.right + 1):
        segments = _column_segments(pixels, pixel_x, plot, spec)
        if not segments:
            samples.append(None)
            continue
        if previous_y is None:
            center, confidence = max(segments, key=lambda item: item[1])
        else:
            center, confidence = max(
                segments,
                key=lambda item: (
                    item[1] - smoothness * abs(item[0] - previous_y) / plot.height
                ),
            )
        previous_y = center
        observed_columns += 1
        samples.append((center, confidence, True))
    samples = _interpolate_short_gaps(samples, maximum_gap)

    points = []
    for offset, sample in enumerate(samples):
        if sample is None:
            continue
        pixel_y, confidence, observed = sample
        pixel_x = plot.left + offset
        points.append(
            Point(
                x=calibration.x_value(pixel_x, plot),
                y=calibration.y_value(pixel_y, plot),
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                confidence=confidence,
                observed=observed,
            )
        )
    return Trace(
        name=spec.name,
        requested_color=format_color(spec.color),
        points=tuple(points),
        observed_columns=observed_columns,
        plot_columns=plot.width + 1,
    )


def _color_distance(
    left: tuple[int, int, int], right: tuple[int, int, int]
) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def _legend_series_specs(
    image: Image.Image,
    plot: PlotBox,
    *,
    maximum_series: int,
    minimum_chroma: int,
) -> list[SeriesSpec]:
    """Read the ordered color swatches from a compact in-plot legend."""

    rgb = image.convert("RGB")
    pixels = rgb.load()
    x_start = plot.left + max(2, round(plot.width * 0.015))
    x_stop = min(plot.right, plot.left + round(plot.width * 0.45))
    y_start = plot.top + 2
    y_stop = min(plot.bottom, plot.top + round(plot.height * 0.44))
    minimum_run = max(10, round(plot.width * 0.025))
    maximum_run = max(minimum_run + 1, round(plot.width * 0.16))
    records: list[tuple[int, int, int, tuple[int, int, int], int]] = []

    for y in range(y_start, y_stop):
        occupied = []
        for x in range(x_start, x_stop):
            color = pixels[x, y]
            chroma = max(color) - min(color)
            luminance = sum(color) / 3
            if chroma >= max(20, minimum_chroma * 2) or (
                chroma <= 12 and 85 <= luminance <= 205
            ):
                occupied.append(x)
        for group in _group_adjacent(occupied):
            width = group[-1] - group[0] + 1
            if not minimum_run <= width <= maximum_run:
                continue
            colors = [pixels[x, y] for x in group]
            middle = len(colors) // 2
            median_color = tuple(
                sorted(color[channel] for color in colors)[middle]
                for channel in range(3)
            )
            core = [
                color
                for color in colors
                if _color_distance(color, median_color) <= 45
            ]
            if len(core) < len(colors) * 0.70:
                continue
            mean_color = tuple(
                round(sum(color[channel] for color in core) / len(core))
                for channel in range(3)
            )
            if max(mean_color) < 55 or min(mean_color) > 235:
                continue
            records.append((y, group[0], group[-1], mean_color, len(core)))

    clusters: list[
        list[tuple[int, int, int, tuple[int, int, int], int]]
    ] = []
    for record in records:
        center = (record[1] + record[2]) / 2
        cluster = next(
            (
                candidate
                for candidate in reversed(clusters)
                if record[0] - candidate[-1][0] <= 4
                and abs(center - (candidate[-1][1] + candidate[-1][2]) / 2) <= 9
            ),
            None,
        )
        if cluster is None:
            clusters.append([record])
        else:
            cluster.append(record)

    representatives = []
    for cluster in clusters:
        if len({record[0] for record in cluster}) < 2:
            continue

        def quality(
            record: tuple[int, int, int, tuple[int, int, int], int],
        ) -> float:
            color = record[3]
            chroma = max(color) - min(color)
            return chroma * 1.5 + 255 - sum(color) / 3 + record[4]

        representatives.append(max(cluster, key=quality))

    if not representatives:
        return []
    alignment_tolerance = max(10, round(plot.width * 0.03))
    aligned_groups = []
    for anchor in representatives:
        anchor_center = (anchor[1] + anchor[2]) / 2
        aligned_groups.append(
            [
                record
                for record in representatives
                if abs((record[1] + record[2]) / 2 - anchor_center)
                <= alignment_tolerance
            ]
        )
    aligned = max(
        aligned_groups,
        key=lambda group: (len(group), sum(record[4] for record in group)),
    )
    # Two incidental horizontal curve fragments are common in unlegended
    # response plots.  Three aligned swatches are a reliable legend signal;
    # smaller legends remain covered by the hue-based fallback below.
    if len(aligned) < 3:
        return []

    specs = []
    for number, record in enumerate(sorted(aligned)[:maximum_series], start=1):
        color = record[3]
        neutral = max(color) - min(color) < 18
        specs.append(
            SeriesSpec(
                name=f"series_{number}",
                color=color,
                color_tolerance=45.0 if neutral else 65.0,
                minimum_chroma=0 if neutral else minimum_chroma,
            )
        )
    return specs


def detect_series_specs(
    image: Image.Image,
    plot: PlotBox,
    *,
    maximum_series: int = 8,
    minimum_chroma: int = 8,
) -> list[SeriesSpec]:
    """Infer series from legend swatches, falling back to plot hue peaks."""

    legend_specs = _legend_series_specs(
        image,
        plot,
        maximum_series=maximum_series,
        minimum_chroma=minimum_chroma,
    )
    if legend_specs:
        return legend_specs

    rgb = image.convert("RGB")
    pixels = rgb.load()
    bins: list[list[tuple[int, int, int]]] = [[] for _ in range(72)]
    detection_chroma = max(20, minimum_chroma)
    for y in range(plot.top + 1, plot.bottom):
        for x in range(plot.left + 1, plot.right):
            color = pixels[x, y]
            if max(color) - min(color) < detection_chroma:
                continue
            hue, saturation, value = colorsys.rgb_to_hsv(
                *(component / 255 for component in color)
            )
            if saturation < 0.12 or value < 0.15:
                continue
            bins[int(hue * len(bins)) % len(bins)].append(color)
    minimum_support = max(8, round(plot.width * 0.025))
    ranked = sorted(range(len(bins)), key=lambda index: len(bins[index]), reverse=True)
    selected: list[int] = []
    for index in ranked:
        if len(bins[index]) < minimum_support:
            break
        if any(
            min(
                (index - other) % len(bins),
                (other - index) % len(bins),
            )
            <= 2
            for other in selected
        ):
            continue
        selected.append(index)
        if len(selected) >= maximum_series:
            break
    specs = []
    for number, index in enumerate(selected, start=1):
        colors = bins[index]
        strongest = sorted(
            colors, key=lambda color: max(color) - min(color), reverse=True
        )[: max(1, len(colors) // 3)]
        mean_color = tuple(
            round(sum(color[channel] for color in strongest) / len(strongest))
            for channel in range(3)
        )
        specs.append(
            SeriesSpec(
                name=f"series_{number}",
                color=mean_color,
                color_tolerance=65.0,
                minimum_chroma=detection_chroma,
            )
        )
    if not specs:
        specs.append(SeriesSpec(name="series_1", minimum_chroma=minimum_chroma))
    return specs


def extract_line_chart(
    image: Image.Image,
    recognizer: TickRecognizer,
    *,
    plot: PlotBox | None = None,
    series_specs: Sequence[SeriesSpec] | None = None,
    x_scale: str = "auto",
    y_scale: str = "auto",
    maximum_series: int = 8,
    minimum_chroma: int = 8,
) -> tuple[PlotBox, TickCalibration, list[Trace]]:
    plot, calibration = infer_tick_calibration(
        image, recognizer, plot=plot, x_scale=x_scale, y_scale=y_scale
    )
    if series_specs is None:
        series_specs = detect_series_specs(
            image,
            plot,
            maximum_series=maximum_series,
            minimum_chroma=minimum_chroma,
        )
    traces = [trace_series(image, plot, calibration, spec) for spec in series_specs]
    return plot, calibration, traces


def find_x_at_y(
    trace: Trace,
    target_y: float,
    calibration: CalibrationLike,
    plot: PlotBox,
    *,
    direction: str = "falling",
    confirmation_points: int = 5,
) -> float:
    if direction not in {"falling", "rising", "either"}:
        raise ChartError(f"unsupported crossing direction: {direction}")
    target_pixel = calibration.y_pixel(target_y, plot)
    requested_color = (
        parse_color(trace.requested_color) if trace.requested_color else None
    )
    neutral_trace = requested_color is not None and (
        max(requested_color) - min(requested_color) < 18
    )
    neutral_candidates: list[tuple[tuple[float, float, int], float]] = []
    for index, (left, right) in enumerate(zip(trace.points, trace.points[1:])):
        if right.pixel_x - left.pixel_x > 8:
            continue
        delta = right.pixel_y - left.pixel_y
        falling = left.pixel_y <= target_pixel <= right.pixel_y and delta > 0
        rising = right.pixel_y <= target_pixel <= left.pixel_y and delta < 0
        if not (
            (direction == "falling" and falling)
            or (direction == "rising" and rising)
            or (direction == "either" and (falling or rising))
        ):
            continue
        confirmed = True
        if confirmation_points:
            post = trace.points[index + 1 : index + 1 + confirmation_points]
            threshold = math.ceil(len(post) * 0.70)
            confirmed = not (
                len(post) < min(3, confirmation_points)
                or (
                    direction == "falling"
                    and sum(point.pixel_y >= target_pixel for point in post)
                    < threshold
                )
                or (
                    direction == "rising"
                    and sum(point.pixel_y <= target_pixel for point in post)
                    < threshold
                )
            )
        high_confidence_vertical = (
            min(left.confidence, right.confidence)
            >= (0.75 if neutral_trace else 0.85)
            and abs(delta) <= plot.height * 0.20
        )
        if not confirmed and not high_confidence_vertical:
            continue
        fraction = (target_pixel - left.pixel_y) / delta
        pixel_x = left.pixel_x + fraction * (right.pixel_x - left.pixel_x)
        value = calibration.x_value(pixel_x, plot)
        if not neutral_trace:
            return value
        if abs(delta) > plot.height * 0.30:
            continue
        neutral_candidates.append(
            (
                (
                    min(left.confidence, right.confidence),
                    (left.confidence + right.confidence) / 2,
                    int(confirmed),
                ),
                value,
            )
        )
    if neutral_candidates:
        return max(neutral_candidates, key=lambda item: item[0])[1]
    raise ChartError(
        f"series {trace.name!r} does not cross y={target_y:g} toward {direction}"
    )


def _fitted_axis_to_json(axis: FittedAxis, first_pixel: int, last_pixel: int) -> dict:
    return {
        "scale": axis.scale,
        "minimum": min(
            axis.pixel_to_value(first_pixel), axis.pixel_to_value(last_pixel)
        ),
        "maximum": max(
            axis.pixel_to_value(first_pixel), axis.pixel_to_value(last_pixel)
        ),
        "slope": axis.slope,
        "intercept": axis.intercept,
        "normalized_rmse": axis.normalized_rmse,
        "ticks": [asdict(tick) for tick in axis.ticks],
    }


def trace_to_json(trace: Trace, *, point_stride: int = 1) -> dict:
    point_stride = max(1, point_stride)
    return {
        "name": trace.name,
        "requested_color": trace.requested_color,
        "coverage": trace.coverage,
        "observed_columns": trace.observed_columns,
        "plot_columns": trace.plot_columns,
        "points": [asdict(point) for point in trace.points[::point_stride]],
    }


def extraction_to_json(
    plot: PlotBox,
    calibration: TickCalibration,
    traces: Sequence[Trace],
    *,
    source: str | None = None,
    point_stride: int = 1,
) -> dict:
    result = {
        "schema_version": 1,
        "type": "line_chart",
        "plot": asdict(plot),
        "axes": {
            "x": _fitted_axis_to_json(calibration.x, plot.left, plot.right),
            "y": _fitted_axis_to_json(calibration.y, plot.bottom, plot.top),
        },
        "series": [trace_to_json(trace, point_stride=point_stride) for trace in traces],
    }
    if source is not None:
        result["source"] = source
    return result


def prepare_ocr_crop(crop: Image.Image, scale: int = 4) -> Image.Image:
    """Upscale a tiny tick label without discarding anti-aliased glyphs."""

    gray = ImageOps.autocontrast(crop.convert("L"))
    gray = ImageOps.expand(gray, border=3, fill=255)
    return gray.resize(
        (gray.width * max(1, scale), gray.height * max(1, scale)),
        Image.Resampling.LANCZOS,
    )


def write_overlay(
    path: str, image: Image.Image, plot: PlotBox, traces: Sequence[Trace]
) -> None:
    overlay = image.convert("RGB").copy()
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(
        (plot.left, plot.top, plot.right, plot.bottom),
        outline=(255, 0, 90),
        width=2,
    )
    palette = [(255, 0, 90), (0, 140, 255), (0, 180, 100), (255, 130, 0)]
    for index, trace in enumerate(traces):
        color = palette[index % len(palette)]
        run: list[tuple[int, int]] = []
        previous_x: int | None = None
        previous_y: float | None = None
        for point in trace.points:
            if previous_x is not None and (
                point.pixel_x - previous_x > 8
                or (
                    previous_y is not None
                    and abs(point.pixel_y - previous_y) > plot.height * 0.15
                )
            ):
                if len(run) > 1:
                    draw.line(run, fill=color, width=1)
                run = []
            run.append((point.pixel_x, round(point.pixel_y)))
            previous_x = point.pixel_x
            previous_y = point.pixel_y
        if len(run) > 1:
            draw.line(run, fill=color, width=1)
    overlay.save(path)
