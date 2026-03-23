"""Tests for CVAT dataset inspection and normalization logic."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from yolo_edge.data.dataset_manager import DatasetManager
from yolo_edge.data.dataset_manager import DatasetValidationError


class DatasetManagerTestCase(unittest.TestCase):
    """Validate dataset inspection and normalization behavior."""

    def setUp(self) -> None:
        """Create shared test fixtures."""
        self.manager = DatasetManager()
        self.dataset_root = PROJECT_ROOT / "data" / "cvat_exports" / "caton_hause"

    def test_inspect_dataset_directory_reports_missing_labels(self) -> None:
        """The extracted CVAT export should report the known missing labels."""
        description = self.manager.inspect_dataset_directory(self.dataset_root)

        self.assertEqual(description.image_count, 74)
        self.assertEqual(description.label_count, 65)
        self.assertEqual(len(description.missing_labels), 9)
        self.assertIn("emrah_carton_hause_71", description.missing_labels)

    def test_prepare_cvat_export_rejects_missing_labels(self) -> None:
        """Normalization should fail by default when annotations are incomplete."""
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory)

            with self.assertRaises(DatasetValidationError):
                self.manager.prepare_cvat_export(
                    dataset_root=self.dataset_root,
                    normalized_dataset_directory=temporary_path / "normalized",
                    image_source_directory=PROJECT_ROOT / "data" / "images",
                    overwrite=True,
                )


if __name__ == "__main__":
    unittest.main()
