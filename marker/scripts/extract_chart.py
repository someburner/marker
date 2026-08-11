"""Command-line extraction and validation for raster line charts."""

from __future__ import annotations

import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import click
from PIL import Image

from marker.charts.line import (
    Axis,
    Calibration,
    ChartError,
    PlotBox,
    SeriesSpec,
    detect_plot_box,
    detect_series_specs,
    extraction_to_json,
    find_x_at_y,
    parse_color,
    trace_series,
    trace_to_json,
    write_overlay,
)
from marker.charts.tesseract import TesseractTickRecognizer
from marker.logger import configure_logging

configure_logging()


def _recognizer(engine: str):
    if engine == "tesseract":
        return TesseractTickRecognizer(), None
    from marker.models import create_model_dict
    from marker.processors.line_chart import SuryaTickRecognizer

    models = create_model_dict()
    return SuryaTickRecognizer(models["recognition_model"]), models


def _shutdown_models(models: dict | None) -> None:
    if models is None:
        return
    from marker.models import shutdown_models

    shutdown_models(models)


def _plot(value: str, image: Image.Image) -> PlotBox:
    if value == "auto":
        return detect_plot_box(image)
    try:
        left, top, right, bottom = (int(part) for part in value.split(","))
    except ValueError as exc:
        raise ChartError("--plot must be auto or left,top,right,bottom") from exc
    plot = PlotBox(left, top, right, bottom)
    plot.validate(*image.size)
    return plot


def _series(
    values: Sequence[str],
    image: Image.Image,
    plot: PlotBox,
    minimum_chroma: int,
    color_tolerance: float,
) -> list[SeriesSpec]:
    if not values:
        return detect_series_specs(image, plot, minimum_chroma=minimum_chroma)
    specs = []
    for value in values:
        if "=" in value:
            name, color = value.split("=", 1)
            parsed_color = parse_color(color)
        else:
            name, parsed_color = value, None
        if not name:
            raise ChartError("series arguments require a non-empty name")
        specs.append(
            SeriesSpec(
                name=name,
                color=parsed_color,
                color_tolerance=color_tolerance,
                minimum_chroma=minimum_chroma,
            )
        )
    return specs


def _manual_json(
    source: str,
    plot: PlotBox,
    calibration: Calibration,
    traces,
    point_stride: int,
) -> dict:
    return {
        "schema_version": 1,
        "type": "line_chart",
        "source": source,
        "plot": asdict(plot),
        "axes": {
            "x": asdict(calibration.x),
            "y": asdict(calibration.y),
        },
        "series": [trace_to_json(trace, point_stride=point_stride) for trace in traces],
    }


def _write_json(destination: str, data: dict) -> None:
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if destination == "-":
        sys.stdout.write(payload)
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def _write_csv(path: Path, traces) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["series", "x", "y", "pixel_x", "pixel_y", "confidence", "observed"]
        )
        for trace in traces:
            for point in trace.points:
                writer.writerow(
                    [
                        trace.name,
                        f"{point.x:.12g}",
                        f"{point.y:.12g}",
                        point.pixel_x,
                        f"{point.pixel_y:.6f}",
                        f"{point.confidence:.6f}",
                        str(point.observed).lower(),
                    ]
                )


@click.group(help="Extract calibrated data series from raster line charts.")
def extract_chart_cli():
    pass


@extract_chart_cli.command("extract")
@click.argument(
    "image_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--ocr-engine", type=click.Choice(["surya", "tesseract"]), default="surya"
)
@click.option(
    "--x-scale", type=click.Choice(["auto", "linear", "log10"]), default="auto"
)
@click.option(
    "--y-scale", type=click.Choice(["auto", "linear", "log10"]), default="auto"
)
@click.option("--x-min", type=float)
@click.option("--x-max", type=float)
@click.option("--y-min", type=float)
@click.option("--y-max", type=float)
@click.option(
    "--plot", "plot_value", default="auto", help="auto or left,top,right,bottom"
)
@click.option("--series", "series_values", multiple=True, metavar="NAME[=#RRGGBB]")
@click.option("--minimum-chroma", type=int, default=5, show_default=True)
@click.option("--color-tolerance", type=float, default=90.0, show_default=True)
@click.option("--point-stride", type=int, default=1, show_default=True)
@click.option("--output", default="-", help="JSON path or - for stdout")
@click.option("--csv-output", type=click.Path(path_type=Path))
@click.option("--overlay", type=click.Path(path_type=Path))
def extract_command(
    image_path: Path,
    ocr_engine: str,
    x_scale: str,
    y_scale: str,
    x_min: float | None,
    x_max: float | None,
    y_min: float | None,
    y_max: float | None,
    plot_value: str,
    series_values: Sequence[str],
    minimum_chroma: int,
    color_tolerance: float,
    point_stride: int,
    output: str,
    csv_output: Path | None,
    overlay: Path | None,
):
    """Extract one chart image to structured JSON and optional CSV."""

    models = None
    try:
        image = Image.open(image_path).convert("RGB")
        plot = _plot(plot_value, image)
        series_specs = _series(
            series_values, image, plot, minimum_chroma, color_tolerance
        )
        bounds = (x_min, x_max, y_min, y_max)
        if any(value is not None for value in bounds):
            if any(value is None for value in bounds):
                raise ChartError(
                    "manual calibration requires --x-min, --x-max, --y-min, and --y-max"
                )
            calibration = Calibration(
                x=Axis(x_min, x_max, "linear" if x_scale == "auto" else x_scale),
                y=Axis(y_min, y_max, "linear" if y_scale == "auto" else y_scale),
            )
            traces = [
                trace_series(image, plot, calibration, spec) for spec in series_specs
            ]
            data = _manual_json(
                str(image_path), plot, calibration, traces, max(1, point_stride)
            )
        else:
            recognizer, models = _recognizer(ocr_engine)
            from marker.charts.line import infer_tick_calibration

            plot, calibration = infer_tick_calibration(
                image,
                recognizer,
                plot=plot,
                x_scale=x_scale,
                y_scale=y_scale,
            )
            traces = [
                trace_series(image, plot, calibration, spec) for spec in series_specs
            ]
            data = extraction_to_json(
                plot,
                calibration,
                traces,
                source=str(image_path),
                point_stride=max(1, point_stride),
            )
        _write_json(output, data)
        if csv_output:
            _write_csv(csv_output, traces)
        if overlay:
            overlay.parent.mkdir(parents=True, exist_ok=True)
            write_overlay(str(overlay), image, plot, traces)
    except (ChartError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        _shutdown_models(models)


def _deep_merge(defaults: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(defaults)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _series_from_manifest(data: dict) -> SeriesSpec:
    color = data.get("color")
    return SeriesSpec(
        name=data["name"],
        color=parse_color(color) if color else None,
        color_tolerance=float(data.get("color_tolerance", 90.0)),
        minimum_chroma=int(data.get("minimum_chroma", 5)),
    )


@extract_chart_cli.command("validate")
@click.argument(
    "manifest_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--ocr-engine", type=click.Choice(["surya", "tesseract"]), default="surya"
)
@click.option(
    "--x-scale", type=click.Choice(["auto", "linear", "log10"]), default="auto"
)
@click.option(
    "--y-scale", type=click.Choice(["auto", "linear", "log10"]), default="auto"
)
@click.option("--output", default="-", help="JSON path or - for stdout")
@click.option("--overlay-dir", type=click.Path(path_type=Path))
def validate_command(
    manifest_path: Path,
    ocr_engine: str,
    x_scale: str,
    y_scale: str,
    output: str,
    overlay_dir: Path | None,
):
    """Validate OCR-calibrated crossings against a checked reference manifest."""

    models = None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        charts = manifest.get("charts")
        if manifest.get("schema_version") != 1 or not isinstance(charts, list):
            raise ChartError(
                "validation manifest must use schema_version 1 and contain charts"
            )
        recognizer, models = _recognizer(ocr_engine)
        defaults = manifest.get("defaults", {})
        results = []
        for override in charts:
            chart = _deep_merge(defaults, override)
            image_path = (manifest_path.parent / chart["image"]).resolve()
            image = Image.open(image_path).convert("RGB")
            plot = _plot("auto", image)
            from marker.charts.line import infer_tick_calibration

            plot, calibration = infer_tick_calibration(
                image,
                recognizer,
                plot=plot,
                x_scale=x_scale,
                y_scale=y_scale,
            )
            specs = [_series_from_manifest(item) for item in chart["series"]]
            traces = [trace_series(image, plot, calibration, spec) for spec in specs]
            trace_by_name = {trace.name: trace for trace in traces}
            references = []
            for reference in chart.get("references", []):
                if reference.get("kind") != "x_at_y":
                    raise ChartError(
                        f"unsupported reference kind: {reference.get('kind')}"
                    )
                expected = float(reference["expected_x"])
                extracted = find_x_at_y(
                    trace_by_name[reference["series"]],
                    float(reference["y"]),
                    calibration,
                    plot,
                    direction=reference.get("direction", "falling"),
                )
                relative_error = abs(extracted - expected) / abs(expected)
                tolerance = float(reference.get("relative_tolerance", 0.10))
                references.append(
                    {
                        "kind": "x_at_y",
                        "series": reference["series"],
                        "y": float(reference["y"]),
                        "expected_x": expected,
                        "extracted_x": extracted,
                        "relative_error": relative_error,
                        "relative_tolerance": tolerance,
                        "passed": relative_error <= tolerance,
                        "source": reference.get("source"),
                    }
                )
            chart_id = chart["id"]
            if overlay_dir:
                overlay_dir.mkdir(parents=True, exist_ok=True)
                write_overlay(str(overlay_dir / f"{chart_id}.png"), image, plot, traces)
            results.append(
                {
                    "id": chart_id,
                    "plot": asdict(plot),
                    "axes": {
                        "x": {
                            "scale": calibration.x.scale,
                            "minimum": calibration.x.pixel_to_value(plot.left),
                            "maximum": calibration.x.pixel_to_value(plot.right),
                        },
                        "y": {
                            "scale": calibration.y.scale,
                            "minimum": calibration.y.pixel_to_value(plot.bottom),
                            "maximum": calibration.y.pixel_to_value(plot.top),
                        },
                    },
                    "references": references,
                    "passed": bool(references)
                    and all(reference["passed"] for reference in references),
                }
            )
        references = [item for result in results for item in result["references"]]
        report = {
            "schema_version": 1,
            "manifest": str(manifest_path.resolve()),
            "charts": results,
            "summary": {
                "charts": len(results),
                "references": len(references),
                "passed_references": sum(item["passed"] for item in references),
                "maximum_relative_error": max(
                    (item["relative_error"] for item in references), default=None
                ),
                "mean_relative_error": (
                    sum(item["relative_error"] for item in references) / len(references)
                    if references
                    else None
                ),
                "passed": bool(references)
                and all(item["passed"] for item in references),
            },
        }
        _write_json(output, report)
        if not report["summary"]["passed"]:
            raise click.exceptions.Exit(1)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"invalid validation manifest: {exc}") from exc
    except (ChartError, OSError, KeyError, TypeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    finally:
        _shutdown_models(models)


if __name__ == "__main__":
    extract_chart_cli()
