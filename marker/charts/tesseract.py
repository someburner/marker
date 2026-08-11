"""Optional Tesseract adapter for the standalone line-chart command."""

from __future__ import annotations

import io
import re
import shutil
import subprocess
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image

from marker.charts.line import (
    ChartError,
    OCRResult,
    TickCrop,
    prepare_ocr_crop,
)


@dataclass(frozen=True)
class _Component:
    size: int
    left: int
    top: int
    right: int
    bottom: int


def _components(image: Image.Image, threshold: int = 210) -> list[_Component]:
    gray = image.convert("L")
    pixels = gray.load()
    seen: set[tuple[int, int]] = set()
    components: list[_Component] = []
    for y in range(gray.height):
        for x in range(gray.width):
            if (x, y) in seen or pixels[x, y] >= threshold:
                continue
            stack = [(x, y)]
            seen.add((x, y))
            points = []
            while stack:
                current_x, current_y = stack.pop()
                points.append((current_x, current_y))
                for neighbor_x in range(
                    max(0, current_x - 1), min(gray.width, current_x + 2)
                ):
                    for neighbor_y in range(
                        max(0, current_y - 1), min(gray.height, current_y + 2)
                    ):
                        if (neighbor_x, neighbor_y) not in seen and pixels[
                            neighbor_x, neighbor_y
                        ] < threshold:
                            seen.add((neighbor_x, neighbor_y))
                            stack.append((neighbor_x, neighbor_y))
            if len(points) >= 2:
                components.append(
                    _Component(
                        size=len(points),
                        left=min(point[0] for point in points),
                        top=min(point[1] for point in points),
                        right=max(point[0] for point in points) + 1,
                        bottom=max(point[1] for point in points) + 1,
                    )
                )
    return components


class TesseractTickRecognizer:
    """Recognize small tick crops without making Tesseract a Marker dependency."""

    def __init__(self, executable: str = "tesseract", scale: int = 8):
        resolved = shutil.which(executable)
        if resolved is None:
            raise ChartError(
                "Tesseract tick OCR was requested, but the tesseract binary was not found"
            )
        self.executable = resolved
        self.scale = scale

    def _run(
        self,
        image: Image.Image,
        *,
        page_segmentation: int,
        whitelist: str | None = None,
        scale: int | None = None,
    ) -> str:
        prepared = prepare_ocr_crop(image, scale or self.scale)
        payload = io.BytesIO()
        prepared.save(payload, format="PNG")
        command = [
            self.executable,
            "stdin",
            "stdout",
            "--psm",
            str(page_segmentation),
        ]
        if whitelist:
            command.extend(["-c", f"tessedit_char_whitelist={whitelist}"])
        result = subprocess.run(
            command,
            input=payload.getvalue(),
            capture_output=True,
            check=False,
        )
        if result.returncode:
            message = result.stderr.decode("utf-8", errors="replace").strip()
            raise ChartError(f"Tesseract tick OCR failed: {message}")
        return result.stdout.decode("utf-8", errors="replace").strip()

    @staticmethod
    def _center_x_label(
        crop: TickCrop,
    ) -> tuple[Image.Image, Image.Image, _Component | None]:
        center = round(crop.pixel - crop.bbox[0])
        half_width = max(12, crop.image.height * 2)
        left = max(0, center - half_width)
        right = min(crop.image.width, center + half_width + 1)
        image = crop.image.crop((left, 0, right, crop.image.height)).convert("L")
        all_components = _components(image)
        components = [
            component
            for component in all_components
            if component.bottom - component.top >= image.height * 0.40
        ]
        if not components:
            return image, image, None
        local_center = center - left
        near = [
            component
            for component in components
            if component.left <= local_center + image.height
            and component.right >= local_center - image.height
        ]
        if not near:
            return image, image, None
        baseline_left = min(component.left for component in near)
        baseline_right = max(component.right for component in near)
        baseline_bottom = max(component.bottom for component in near)
        baseline_width = baseline_right - baseline_left
        superscripts = [
            component
            for component in all_components
            if component not in near
            and component.left >= baseline_left + baseline_width * 0.55
            and component.left <= baseline_right + 6
            and component.bottom <= baseline_bottom
        ]
        near.extend(superscripts)
        bbox = _Component(
            size=sum(component.size for component in near),
            left=min(component.left for component in near),
            top=min(component.top for component in near),
            right=max(component.right for component in near),
            bottom=max(component.bottom for component in near),
        )
        tight = image.crop(
            (
                max(0, bbox.left - 3),
                max(0, bbox.top - 2),
                min(image.width, bbox.right + 3),
                min(image.height, bbox.bottom + 2),
            )
        )
        return tight, image, bbox

    def _power_votes(self, image: Image.Image, bbox: _Component | None) -> list[int]:
        if bbox is None or bbox.right - bbox.left < 6:
            return []
        gray = image.convert("L")
        pixels = gray.load()
        width = bbox.right - bbox.left
        split = bbox.left + round(width * 0.70)
        left_pixels = [
            (x, y)
            for y in range(gray.height)
            for x in range(bbox.left, min(split, gray.width))
            if pixels[x, y] < 210
        ]
        right_pixels = [
            (x, y)
            for y in range(gray.height)
            for x in range(max(0, split), min(bbox.right, gray.width))
            if pixels[x, y] < 210
        ]
        if not left_pixels or not right_pixels:
            return []
        if max(y for _, y in right_pixels) > max(y for _, y in left_pixels) - 2:
            return []

        votes = []
        for fraction, page_segmentation in (
            (0.72, 13),
            (0.68, 10),
            (0.60, 10),
            (0.58, 13),
        ):
            exponent = image.crop(
                (
                    bbox.left + round(width * fraction),
                    max(0, bbox.top - 2),
                    min(image.width, bbox.right + 4),
                    min(
                        image.height, bbox.top + round((bbox.bottom - bbox.top) * 0.75)
                    ),
                )
            )
            text = self._run(
                exponent,
                page_segmentation=page_segmentation,
                whitelist="0123456789",
                scale=max(16, self.scale),
            )
            match = re.fullmatch(r"\s*(\d)\s*", text)
            if match:
                votes.append(int(match.group(1)))
        return votes

    def recognize(self, crops: Sequence[TickCrop]) -> Sequence[OCRResult]:
        results = []
        x_result_indices = []
        x_power_votes: list[list[int]] = []
        for crop in crops:
            if crop.axis == "x":
                x_result_indices.append(len(results))
                centered, source, bbox = self._center_x_label(crop)
                if bbox is None:
                    x_power_votes.append([])
                    results.append(OCRResult(""))
                    continue
                text = self._run(centered, page_segmentation=7)
                # Tesseract commonly reads small superscripts as punctuation.
                # The second pass isolates the raised final glyph.
                x_power_votes.append(self._power_votes(source, bbox))
            else:
                text = self._run(crop.image, page_segmentation=7)
                text = re.sub(r"^[‘’“”'\"]+\s*(?=\d)", "-", text)
                if re.fullmatch(r"[oO0]+", text):
                    text = "0"
            results.append(OCRResult(text=text))

        # Power-of-ten labels are centered on consecutive decade grid lines.
        # Candidate superscripts vote for the one consecutive-decade sequence
        # that best explains every label, making one bad tiny glyph harmless.
        base_candidates = {
            exponent - ordinal
            for ordinal, votes in enumerate(x_power_votes)
            for exponent in votes
        }
        if base_candidates:
            vote_counts = [Counter(votes) for votes in x_power_votes]
            base_exponent = max(
                base_candidates,
                key=lambda base: (
                    sum(
                        counts.get(base + ordinal, 0)
                        for ordinal, counts in enumerate(vote_counts)
                    ),
                    sum(
                        base + ordinal in counts
                        for ordinal, counts in enumerate(vote_counts)
                    ),
                    -abs(base),
                ),
            )
            for ordinal, result_index in enumerate(x_result_indices):
                text = results[result_index].text
                if re.search(r"[1lI][0oO]", text) or text.startswith("10^("):
                    results[result_index] = OCRResult(
                        text=f"10^({base_exponent + ordinal})"
                    )
        return results
