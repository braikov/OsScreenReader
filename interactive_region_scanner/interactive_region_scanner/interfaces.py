from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box in screen coordinates."""
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class OcrResult:
    """Recognized text and confidence score."""
    text: str
    confidence: float


@dataclass(frozen=True)
class DetectedRegion:
    """Normalized region with optional text and source metadata."""
    bbox: BoundingBox
    text: str | None
    confidence: float
    source: str


class SessionRepository(ABC):
    """Interface for listing sessions and persisting results."""
    @abstractmethod
    def list_sessions(self) -> Iterable[Path]:
        """Return session directories in processing order."""
        raise NotImplementedError

    @abstractmethod
    def has_result(self, session_path: Path) -> bool:
        """Return True if the session already has a result JSON."""
        raise NotImplementedError

    @abstractmethod
    def load_session_metadata(self, session_path: Path) -> dict:
        """Load session metadata from disk."""
        raise NotImplementedError

    @abstractmethod
    def write_result(self, session_path: Path, payload: dict) -> None:
        """Persist the computed result JSON."""
        raise NotImplementedError

    @abstractmethod
    def write_no_text_result(self, session_path: Path, payload: dict) -> None:
        """Persist regions that lack recognized text."""
        raise NotImplementedError


class FrameProvider(ABC):
    """Interface for accessing and pruning capture frames."""
    @abstractmethod
    def baseline_path(self, session_path: Path) -> Path:
        """Return the baseline screenshot path."""
        raise NotImplementedError

    @abstractmethod
    def frame_paths(self, session_path: Path) -> Sequence[Path]:
        """Return ordered frame paths for the session."""
        raise NotImplementedError

    @abstractmethod
    def delete_frame(self, frame_path: Path) -> None:
        """Remove a processed frame from disk."""
        raise NotImplementedError


class DiffDetector(ABC):
    """Interface for locating changed regions between frames."""
    @abstractmethod
    def find_regions(self, previous_path: Path, current_path: Path) -> list[BoundingBox]:
        """Return bounding boxes that differ between two frames."""
        raise NotImplementedError


class OcrEngine(ABC):
    """Interface for extracting text from a region."""
    @abstractmethod
    def recognize(self, image_path: Path, region: BoundingBox) -> OcrResult | None:
        """Return OCR results for a cropped region, if any."""
        raise NotImplementedError


class RegionDeduplicator(ABC):
    """Interface for merging overlapping detections."""
    @abstractmethod
    def merge(self, regions: list[DetectedRegion]) -> list[DetectedRegion]:
        """Return a deduplicated list of regions."""
        raise NotImplementedError
