from __future__ import annotations

import json
from pathlib import Path

from interactive_region_scanner.processing import (
    EasyOcrEngine,
    FileFrameProvider,
    FileSessionRepository,
    IoUDeduplicator,
    OpenCVDiffDetector,
    process_sessions,
)


def main() -> None:
    """Load configuration and run the interactive region scanner."""
    config_path = Path(__file__).resolve().parents[1] / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    root = Path(config["root"])
    repository = FileSessionRepository(root)
    frame_provider = FileFrameProvider()
    diff_detector = OpenCVDiffDetector(
        threshold=int(config.get("diff_threshold", 30)),
        min_area=int(config.get("min_area", 50)),
    )
    ocr_engine = EasyOcrEngine(
        languages=list(config.get("language", ["en"])),
        use_gpu=bool(config.get("use_gpu", False)),
    )
    deduplicator = IoUDeduplicator(iou_threshold=float(config.get("iou_threshold", 0.5)))

    delete_processed_frames = bool(config.get("delete_frames", False))
    process_sessions(
        repository,
        frame_provider,
        diff_detector,
        ocr_engine,
        deduplicator,
        delete_processed_frames=delete_processed_frames,
    )


if __name__ == "__main__":
    main()
