"""Edge export and lightweight benchmark utilities for Raspberry Pi deployments."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import logging
from pathlib import Path
import shutil
import time

import yaml

from yolo_edge.core.detector import ObjectDetector

try:
    from ultralytics import YOLO
except ModuleNotFoundError as error:
    raise ModuleNotFoundError(
        "Missing Ultralytics dependency while importing the edge export module. "
        "Activate the virtual environment and run 'pip install -r requirements.txt'."
    ) from error


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportArtifact:
    """Describe one exported edge artifact."""

    format_name: str
    artifact_path: Path


class EdgeExporter:
    """Export trained YOLO models into edge-friendly formats and benchmark them."""

    def export_model(
        self,
        model_path: Path,
        export_directory: Path,
        formats: tuple[str, ...],
        image_size: int,
        device: str | None,
        half: bool,
        int8: bool,
        dynamic: bool,
    ) -> tuple[ExportArtifact, ...]:
        """Export a trained YOLO model to one or more deployment formats."""
        resolved_model_path = model_path.resolve()
        if not resolved_model_path.is_file():
            raise FileNotFoundError(f"Export source model not found: {resolved_model_path}")

        export_directory = export_directory.resolve()
        export_directory.mkdir(parents=True, exist_ok=True)

        self._validate_export_dependencies(formats)
        model = YOLO(str(resolved_model_path))
        self._write_labels_file(export_directory=export_directory, model=model)
        artifacts: list[ExportArtifact] = []

        for format_name in formats:
            LOGGER.info("Exporting model format=%s source=%s", format_name, resolved_model_path)
            export_result = model.export(
                format=format_name,
                imgsz=image_size,
                device=device,
                half=half,
                int8=int8,
                dynamic=dynamic,
            )
            exported_path = Path(str(export_result)).resolve()
            target_path = export_directory / exported_path.name
            if exported_path != target_path:
                if exported_path.is_dir():
                    if target_path.exists():
                        shutil.rmtree(target_path)
                    shutil.move(str(exported_path), str(target_path))
                else:
                    shutil.move(str(exported_path), str(target_path))
                exported_path = target_path
            artifacts.append(ExportArtifact(format_name=format_name, artifact_path=exported_path))
            LOGGER.info("Created export artifact format=%s path=%s", format_name, exported_path)

        self._write_export_manifest(
            export_directory=export_directory,
            source_model_path=resolved_model_path,
            artifacts=tuple(artifacts),
            image_size=image_size,
            half=half,
            int8=int8,
            dynamic=dynamic,
        )
        return tuple(artifacts)

    def benchmark_native_model(
        self,
        model_path: Path,
        image_path: Path,
        confidence: float,
        image_size: int,
        device: str | None,
        runs: int,
    ) -> dict[str, float]:
        """Measure lightweight end-to-end inference latency on one reference image."""
        resolved_image_path = image_path.resolve()
        if not resolved_image_path.is_file():
            raise FileNotFoundError(f"Benchmark image not found: {resolved_image_path}")

        detector = ObjectDetector(
            model_path=model_path,
            confidence_threshold=confidence,
            device=device,
            image_size=image_size,
        )
        try:
            import cv2
        except ModuleNotFoundError as error:
            raise ModuleNotFoundError(
                "Missing OpenCV dependency while benchmarking the model."
            ) from error

        frame = cv2.imread(str(resolved_image_path))
        if frame is None:
            raise ValueError(f"Unable to read benchmark image: {resolved_image_path}")

        inference_times_ms: list[float] = []
        for _ in range(runs):
            started_at = time.perf_counter()
            detector.predict(frame)
            inference_times_ms.append((time.perf_counter() - started_at) * 1000.0)

        average_latency = sum(inference_times_ms) / len(inference_times_ms)
        min_latency = min(inference_times_ms)
        max_latency = max(inference_times_ms)
        throughput_fps = 1000.0 / average_latency if average_latency > 0 else 0.0

        benchmark_summary = {
            "runs": float(runs),
            "average_latency_ms": average_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "throughput_fps": throughput_fps,
        }
        LOGGER.info("Native benchmark summary: %s", benchmark_summary)
        return benchmark_summary

    def _write_export_manifest(
        self,
        export_directory: Path,
        source_model_path: Path,
        artifacts: tuple[ExportArtifact, ...],
        image_size: int,
        half: bool,
        int8: bool,
        dynamic: bool,
    ) -> None:
        """Persist a small manifest for edge deployment traceability."""
        manifest_path = export_directory / "edge_export_manifest.yaml"
        manifest_payload = {
            "source_model": str(source_model_path),
            "image_size": image_size,
            "half": half,
            "int8": int8,
            "dynamic": dynamic,
            "artifacts": [
                {
                    "format": artifact.format_name,
                    "path": str(artifact.artifact_path),
                }
                for artifact in artifacts
            ],
            "raspberry_pi_notes": [
                "Prefer the smallest model that still satisfies accuracy requirements.",
                "Validate thermal behavior on Raspberry Pi 5 during continuous inference.",
                "For Raspberry Pi AI Camera workflows, keep a traced export artifact and labels file with the model.",
            ],
        }
        with manifest_path.open("w", encoding="utf-8") as manifest_file:
            yaml.safe_dump(manifest_payload, manifest_file, sort_keys=False)
        LOGGER.info("Saved edge export manifest to %s", manifest_path)

    def _write_labels_file(self, export_directory: Path, model: YOLO) -> None:
        """Write a labels file that can travel with exported edge artifacts."""
        labels_path = export_directory / "labels.txt"
        names = getattr(model, "names", {})
        if isinstance(names, dict):
            ordered_names = [str(value) for _, value in sorted(names.items())]
        elif isinstance(names, (list, tuple)):
            ordered_names = [str(value) for value in names]
        else:
            ordered_names = []

        labels_path.write_text("\n".join(ordered_names) + ("\n" if ordered_names else ""), encoding="utf-8")
        LOGGER.info("Saved labels file to %s", labels_path)

    def _validate_export_dependencies(self, formats: tuple[str, ...]) -> None:
        """Fail early when export dependencies are missing instead of relying on implicit auto-installs."""
        required_modules: set[str] = set()
        normalized_formats = {format_name.lower() for format_name in formats}

        if "onnx" in normalized_formats:
            required_modules.update({"onnx", "onnxruntime", "onnxslim"})
        if "openvino" in normalized_formats:
            required_modules.add("openvino")
        if "tflite" in normalized_formats:
            required_modules.add("tensorflow")

        missing_modules = sorted(
            module_name for module_name in required_modules
            if importlib.util.find_spec(module_name) is None
        )
        if not missing_modules:
            return

        install_hint = self._build_install_hint(missing_modules)
        raise ModuleNotFoundError(
            "Missing export dependencies for the requested formats: "
            f"{missing_modules}. Install them first with: {install_hint}"
        )

    def _build_install_hint(self, missing_modules: list[str]) -> str:
        """Build a deterministic install command for missing export dependencies."""
        requirement_map = {
            "onnx": "onnx>=1.12.0,<=1.19.1",
            "onnxruntime": "onnxruntime",
            "onnxslim": "onnxslim>=0.1.71",
            "openvino": "openvino",
            "tensorflow": "tensorflow",
        }
        packages = [requirement_map[module_name] for module_name in missing_modules]
        return ".venv/bin/pip install " + " ".join(packages)
