from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

from interactive_region_scanner.interfaces import (
    BoundingBox,
    DetectedRegion,
    DiffDetector,
    FrameProvider,
    OcrEngine,
    OcrResult,
    RegionDeduplicator,
    SessionRepository,
)


def process_sessions(
    repository: SessionRepository,
    frame_provider: FrameProvider,
    diff_detector: DiffDetector,
    ocr_engine: OcrEngine,
    deduplicator: RegionDeduplicator,
    delete_processed_frames: bool = True,
) -> None:
    """Run the processing pipeline for all sessions without results."""
    for session_path in repository.list_sessions():
        if repository.has_result(session_path):
            continue

        session_metadata = repository.load_session_metadata(session_path)
        baseline_path = frame_provider.baseline_path(session_path)
        frames = list(frame_provider.frame_paths(session_path))
        total_frames = len(frames)
        elements: list[DetectedRegion] = []

        previous_path = baseline_path
        for index, frame_path in enumerate(frames, start=1):
            frame_start = time.perf_counter()
            regions = diff_detector.find_regions(previous_path, frame_path)
            for region in regions:
                ocr_result = ocr_engine.recognize(frame_path, region)
                if ocr_result is None:
                    elements.append(
                        DetectedRegion(
                            bbox=region,
                            text=None,
                            confidence=0.0,
                            source="hover-diff",
                        )
                    )
                    continue

                elements.append(
                    DetectedRegion(
                        bbox=region,
                        text=ocr_result.text,
                        confidence=ocr_result.confidence,
                        source="hover-diff",
                    )
                )

            if previous_path != baseline_path:
                if delete_processed_frames:
                    frame_provider.delete_frame(previous_path)
            previous_path = frame_path
            elapsed_ms = (time.perf_counter() - frame_start) * 1000
            print(f"{index} of {total_frames} processed in {elapsed_ms:.0f} ms")

        if delete_processed_frames and previous_path != baseline_path:
            frame_provider.delete_frame(previous_path)

        merged = deduplicator.merge(elements)
        numbered: list[tuple[DetectedRegion, str]] = [
            (element, f"elem_{index + 1:04d}") for index, element in enumerate(merged)
        ]

        def has_text(item: DetectedRegion) -> bool:
            """Return True when OCR produced non-empty text."""
            return bool(item.text and item.text.strip())

        with_text = [item for item in numbered if has_text(item[0])]
        without_text = [item for item in numbered if not has_text(item[0])]
        sorted_with_text = sorted(with_text, key=lambda item: item[0].text.strip().lower()) if with_text else []

        def build_payload(items: list[tuple[DetectedRegion, str]]) -> dict:
            return {
                "schema_version": "1.0",
                "session_id": session_metadata.get("session_id"),
                "baseline": baseline_path.name,
                "elements": [
                    {
                        "id": element_id,
                        "bbox": asdict(element.bbox),
                        "text": element.text,
                        "confidence": element.confidence,
                        "source": element.source,
                    }
                    for element, element_id in items
                ],
            }

        payload_with_text = build_payload(sorted_with_text)
        payload_without_text = build_payload(without_text)

        crops_dir = session_path / "result_images"
        crops_dir.mkdir(exist_ok=True)
        with Image.open(baseline_path) as baseline_image:
            for element, element_id in numbered:
                bbox = element.bbox
                crop = baseline_image.crop((bbox.x, bbox.y, bbox.x + bbox.w, bbox.y + bbox.h))
                crop.save(crops_dir / f"{element_id}.png")

        repository.write_result(session_path, payload_with_text)
        repository.write_no_text_result(session_path, payload_without_text)


class FileSessionRepository(SessionRepository):
    """File-backed session repository rooted at a capture directory."""
    def __init__(self, root: Path) -> None:
        """Create a repository rooted at the given directory."""
        self._root = root

    def list_sessions(self) -> Iterable[Path]:
        """Return session directories in ascending order."""
        return sorted(path for path in self._root.iterdir() if path.is_dir())

    def has_result(self, session_path: Path) -> bool:
        """Check whether the session already contains a result.json file."""
        return (session_path / "result.json").exists()

    def load_session_metadata(self, session_path: Path) -> dict:
        """Load session.json metadata, if present."""
        session_file = session_path / "session.json"
        if not session_file.exists():
            return {}
        return json.loads(session_file.read_text(encoding="utf-8"))

    def write_result(self, session_path: Path, payload: dict) -> None:
        """Write the result.json payload for a session."""
        output_path = session_path / "result.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_no_text_result(self, session_path: Path, payload: dict) -> None:
        """Write the result.notext.json payload for regions without recognized text."""
        output_path = session_path / "result.notext.json"
        output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class FileFrameProvider(FrameProvider):
    """Frame provider that reads and deletes files from disk."""
    def baseline_path(self, session_path: Path) -> Path:
        """Return the baseline screenshot path."""
        return session_path / "baseline.png"

    def frame_paths(self, session_path: Path) -> list[Path]:
        """Return ordered frame paths in the frames directory."""
        frames_dir = session_path / "frames"
        if not frames_dir.exists():
            return []
        return sorted(frames_dir.glob("frame_*.png"))

    def delete_frame(self, frame_path: Path) -> None:
        """Delete a processed frame file if it exists."""
        if frame_path.exists():
            frame_path.unlink()


class OpenCVDiffDetector(DiffDetector):
    """Diff detector based on OpenCV absolute differences."""
    def __init__(self, threshold: int = 30, min_area: int = 50) -> None:
        """Initialize with thresholding and minimum area filters."""
        self._threshold = threshold
        self._min_area = min_area

    def find_regions(self, previous_path: Path, current_path: Path) -> list[BoundingBox]:
        """Compute bounding boxes of changed pixels between frames."""
        previous = cv2.imread(str(previous_path))
        current = cv2.imread(str(current_path))
        if previous is None or current is None:
            return []

        diff = cv2.absdiff(previous, current)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, self._threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions: list[BoundingBox] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h < self._min_area:
                continue
            regions.append(BoundingBox(x=x, y=y, w=w, h=h))
        return regions


class EasyOcrEngine(OcrEngine):
    """OCR engine powered by EasyOCR."""
    def __init__(self, languages: list[str] | None = None, use_gpu: bool = False) -> None:
        """Create an EasyOCR reader with optional GPU usage."""
        import easyocr

        self._reader = easyocr.Reader(languages or ["en"], gpu=use_gpu)

    def recognize(self, image_path: Path, region: BoundingBox) -> OcrResult | None:
        """Run OCR over a cropped region and return the best result."""
        image = Image.open(image_path)
        crop = image.crop((region.x, region.y, region.x + region.w, region.y + region.h))
        crop_array = np.array(crop)
        results = self._reader.readtext(crop_array)
        if not results:
            return None

        best = max(results, key=lambda item: item[2])
        text, confidence = best[1], float(best[2])
        return OcrResult(text=text, confidence=confidence)


class IoUDeduplicator(RegionDeduplicator):
    """Deduplicator based on IoU overlap between regions."""
    def __init__(self, iou_threshold: float = 0.5) -> None:
        """Initialize the IoU threshold for merging."""
        self._iou_threshold = iou_threshold

    def merge(self, regions: list[DetectedRegion]) -> list[DetectedRegion]:
        """Return unique regions based on IoU overlap."""
        merged: list[DetectedRegion] = []
        for region in regions:
            if not self._is_duplicate(region, merged):
                merged.append(region)
        return merged

    def _is_duplicate(self, region: DetectedRegion, existing: list[DetectedRegion]) -> bool:
        """Check whether the region overlaps an existing one."""
        for other in existing:
            if self._iou(region.bbox, other.bbox) >= self._iou_threshold:
                return True
        return False

    @staticmethod
    def _iou(a: BoundingBox, b: BoundingBox) -> float:
        """Compute intersection-over-union between two boxes."""
        x_left = max(a.x, b.x)
        y_top = max(a.y, b.y)
        x_right = min(a.x + a.w, b.x + b.w)
        y_bottom = min(a.y + a.h, b.y + b.h)

        if x_right <= x_left or y_bottom <= y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = a.w * a.h + b.w * b.h - intersection
        return intersection / union if union > 0 else 0.0
