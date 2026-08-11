from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw, ImageFont

from marker.charts.line import (
    ChartError,
    OCRResult,
    SeriesSpec,
    Tick,
    detect_plot_box,
    extract_line_chart,
    find_tick_crops,
    fit_axis,
    parse_tick_value,
)
from marker.processors.line_chart import LineChartProcessor
from marker.renderers.json import JSONRenderer
from marker.schema import BlockTypes
from marker.schema.blocks.figure import Figure
from marker.schema.document import Document
from marker.schema.groups.page import PageGroup
from marker.schema.polygon import PolygonBox

PLOT = (70, 30, 600, 300)
X_TICKS = {70: "10<sup>2</sup>", 250: "10<sup>3</sup>", 430: "10<sup>4</sup>"}
Y_TICKS = {30: "0", 165: "-50", 300: "-100"}


def synthetic_chart():
    image = Image.new("RGB", (640, 350), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    left, top, right, bottom = PLOT
    for y, label in Y_TICKS.items():
        draw.line((left, y, right, y), fill=(230, 230, 230))
        text_bbox = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (left - (text_bbox[2] - text_bbox[0]) - 5, y - 4),
            label,
            fill="black",
            font=font,
        )
    for x, exponent in zip(X_TICKS, (2, 3, 4)):
        draw.line((x, top, x, bottom), fill=(220, 220, 220))
        label = f"10^{exponent}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        draw.text(
            (x - (text_bbox[2] - text_bbox[0]) / 2, bottom + 4),
            label,
            fill="black",
            font=font,
        )
    draw.line((right, top, right, bottom), fill=(200, 200, 200))
    draw.rectangle(PLOT, outline=(200, 200, 200))
    points = []
    for x in range(left, right + 1):
        fraction = (x - left) / (right - left)
        y = top + 8 + round((fraction**5) * (bottom - top - 16) * 0.85)
        points.append((x, y))
    draw.line(points, fill=(40, 145, 195), width=2)
    return image


class StaticTickRecognizer:
    def recognize(self, crops):
        results = []
        for crop in crops:
            expected = X_TICKS if crop.axis == "x" else Y_TICKS
            pixel = min(expected, key=lambda candidate: abs(candidate - crop.pixel))
            if abs(pixel - crop.pixel) > 8:
                results.append(OCRResult(""))
            else:
                results.append(OCRResult(expected[pixel], confidence=0.99))
        return results


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("10<sup>4</sup>", 10000),
        ("10⁻²", 0.01),
        ("2.5 × 10^3 Hz", 2500),
        ("−30 dB", -30),
        ("1.25e-3", 0.00125),
    ],
)
def test_parse_tick_value(text, expected):
    assert parse_tick_value(text) == pytest.approx(expected)


def test_fit_axis_rejects_outlier_and_obeys_direction():
    ticks = [
        Tick("y", 10, 0, "0"),
        Tick("y", 20, -10, "-10"),
        Tick("y", 30, 20, "20"),
        Tick("y", 40, -30, "-30"),
    ]
    axis = fit_axis(ticks, "linear", direction="decreasing")
    assert axis.pixel_to_value(40) == pytest.approx(-30)
    assert [tick.inlier for tick in axis.ticks] == [True, True, False, True]


def test_fit_axis_requires_consistent_ticks():
    with pytest.raises(ChartError, match="fewer than two"):
        fit_axis([Tick("x", 10, 100, "100")], direction="increasing")


def test_tick_ocr_calibrates_and_traces_synthetic_chart():
    image = synthetic_chart()
    plot, calibration, traces = extract_line_chart(
        image,
        StaticTickRecognizer(),
        series_specs=[SeriesSpec("response", minimum_chroma=5)],
    )

    assert plot.left == pytest.approx(PLOT[0], abs=1)
    assert plot.right == pytest.approx(PLOT[2], abs=1)
    assert calibration.x.scale == "log10"
    assert calibration.y.scale == "linear"
    assert calibration.x.pixel_to_value(70) == pytest.approx(100, rel=0.02)
    assert calibration.x.pixel_to_value(430) == pytest.approx(10000, rel=0.02)
    assert calibration.y.pixel_to_value(300) == pytest.approx(-100, abs=1)
    assert traces[0].coverage > 0.80
    assert len(traces[0].points) > 400


def _document_with_figure(image):
    polygon = PolygonBox(
        polygon=[
            [0, 0],
            [image.width, 0],
            [image.width, image.height],
            [0, image.height],
        ]
    )
    figure = Figure(
        polygon=polygon,
        page_id=0,
        block_id=0,
        structure=[],
        highres_image=image,
    )
    page = PageGroup(
        polygon=polygon,
        page_id=0,
        structure=[figure.id],
        children=[figure],
        highres_image=image,
        lowres_image=image,
    )
    return Document(filepath="synthetic.png", pages=[page]), figure


class FakeRecognitionModel:
    disable_tqdm = False

    def __call__(self, images, layout_results, full_page):
        assert full_page is False
        assert len(images) == len(layout_results) == 6
        texts = ["0", "-50", "-100", *X_TICKS.values()]
        return [
            SimpleNamespace(
                blocks=[
                    SimpleNamespace(
                        html=text,
                        confidence=0.98,
                        error=False,
                    )
                ]
            )
            for text in texts
        ]


def test_processor_adds_chart_data_to_json_output():
    document, figure = _document_with_figure(synthetic_chart())
    processor = LineChartProcessor(
        FakeRecognitionModel(),
        config={"extract_line_charts": True, "line_chart_minimum_chroma": 5},
    )
    processor(document)

    assert figure.chart_data["type"] == "line_chart"
    assert figure.chart_data["axes"]["x"]["scale"] == "log10"
    assert figure.chart_data["series"][0]["points"]

    output = JSONRenderer({"extract_images": False})(document)
    figure_output = next(
        child
        for page in output.children
        for child in page.children
        if child.block_type == str(BlockTypes.Figure)
    )
    assert figure_output.chart_data == figure.chart_data


def test_tick_crop_detection_uses_label_positions():
    image = synthetic_chart()
    plot = detect_plot_box(image)
    x_pixels = [crop.pixel for crop in find_tick_crops(image, plot) if crop.axis == "x"]
    assert x_pixels == pytest.approx(list(X_TICKS), abs=3)
