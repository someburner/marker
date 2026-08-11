"""Optional structured extraction for line-chart Figure blocks."""

from collections import Counter
from collections.abc import Sequence
from typing import Annotated

from surya.layout.schema import LayoutBox, LayoutResult
from surya.recognition import RecognitionPredictor

from marker.charts.line import (
    ChartError,
    OCRResult,
    TickCrop,
    extract_line_chart,
    extraction_to_json,
    prepare_ocr_crop,
)
from marker.logger import get_logger
from marker.processors import BaseProcessor
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.labels import block_type_to_surya_label

logger = get_logger()


class SuryaTickRecognizer:
    """Adapt Marker's recognition predictor to small, pre-detected tick labels."""

    def __init__(
        self,
        recognition_model: RecognitionPredictor,
        *,
        disable_tqdm: bool = False,
        scale: int = 4,
    ):
        self.recognition_model = recognition_model
        self.disable_tqdm = disable_tqdm
        self.scale = scale

    def recognize(self, crops: Sequence[TickCrop]) -> Sequence[OCRResult]:
        if not crops:
            return []
        images = [prepare_ocr_crop(crop.image, self.scale) for crop in crops]
        label = block_type_to_surya_label(BlockTypes.Text) or "Text"
        layouts = []
        for image in images:
            width, height = image.size
            box = LayoutBox(
                polygon=[[0, 0], [width, 0], [width, height], [0, height]],
                label=label,
                raw_label=label,
                position=0,
                count=64,
            )
            layouts.append(LayoutResult(bboxes=[box], image_bbox=[0, 0, width, height]))
        self.recognition_model.disable_tqdm = self.disable_tqdm
        page_results = self.recognition_model(
            images=images, layout_results=layouts, full_page=False
        )
        results = []
        for page_result in page_results:
            block = page_result.blocks[0] if page_result.blocks else None
            if block is None or block.error:
                results.append(OCRResult(""))
            else:
                results.append(
                    OCRResult(
                        text=block.html or "",
                        confidence=getattr(block, "confidence", None),
                    )
                )
        return results


class LineChartProcessor(BaseProcessor):
    """Extract calibrated series from Figure blocks when explicitly enabled."""

    block_types = (BlockTypes.Figure,)
    extract_line_charts: Annotated[
        bool,
        "Extract structured line-chart data from Figure blocks with tick OCR.",
    ] = False
    line_chart_x_scale: Annotated[
        str,
        "X-axis scale hint: auto, linear, or log10.",
    ] = "auto"
    line_chart_y_scale: Annotated[
        str,
        "Y-axis scale hint: auto, linear, or log10.",
    ] = "auto"
    line_chart_maximum_series: Annotated[
        int,
        "Maximum number of colored line series detected per chart.",
    ] = 1
    line_chart_minimum_chroma: Annotated[
        int,
        "Minimum RGB channel spread for a plotted line pixel.",
    ] = 5
    line_chart_point_stride: Annotated[
        int,
        "Keep every nth extracted point in structured JSON output.",
    ] = 1
    line_chart_ocr_scale: Annotated[
        int,
        "Upscaling factor applied to small tick labels before OCR.",
    ] = 4
    disable_tqdm: Annotated[
        bool,
        "Disable recognition progress bars.",
    ] = False
    disable_ocr: Annotated[bool, "Disable OCR entirely."] = False

    def __init__(
        self,
        recognition_model: RecognitionPredictor,
        config=None,
    ):
        super().__init__(config)
        self.recognition_model = recognition_model
        self.chart_stats = Counter()

    def __call__(self, document: Document):
        if not self.extract_line_charts or self.disable_ocr:
            return
        figures = document.contained_blocks(self.block_types)
        if not figures:
            return
        recognizer = SuryaTickRecognizer(
            self.recognition_model,
            disable_tqdm=self.disable_tqdm,
            scale=self.line_chart_ocr_scale,
        )
        for block in figures:
            image = block.get_image(document, highres=True)
            if image is None:
                self.chart_stats["missing_image"] += 1
                continue
            try:
                plot, calibration, traces = extract_line_chart(
                    image,
                    recognizer,
                    x_scale=self.line_chart_x_scale,
                    y_scale=self.line_chart_y_scale,
                    maximum_series=max(1, self.line_chart_maximum_series),
                    minimum_chroma=max(0, self.line_chart_minimum_chroma),
                )
            except ChartError as exc:
                self.chart_stats["not_line_chart"] += 1
                logger.debug(f"Line chart extraction skipped for {block.id}: {exc}")
                continue
            block.chart_data = extraction_to_json(
                plot,
                calibration,
                traces,
                source=str(block.id),
                point_stride=max(1, self.line_chart_point_stride),
            )
            self.chart_stats["extracted"] += 1
        self.chart_stats["figures"] += len(figures)
        logger.info(f"Line chart processing stats: {dict(self.chart_stats)}")
