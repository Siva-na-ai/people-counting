"""
IoU-based multi-object tracker inspired by SORT / ByteTrack.

Performs frame-to-frame association of detections using IoU cost matrix
and the Hungarian algorithm. Maintains track state (tentative, confirmed, deleted)
and supports global ID assignment from the ReID gallery.
"""

import logging
from enum import Enum, auto

import numpy as np
from scipy.optimize import linear_sum_assignment

import config
from utils import compute_iou_matrix

logger = logging.getLogger(__name__)


class TrackState(Enum):
    """State of a tracked object."""
    TENTATIVE = auto()   # Not yet confirmed (needs TRACKER_MIN_HITS consecutive hits)
    CONFIRMED = auto()   # Actively tracked
    DELETED = auto()     # Lost for too long, will be removed


class Track:
    """
    A single tracked person.

    Attributes:
        track_id: Unique track ID (local to this tracker session).
        bbox: Current bounding box [x1, y1, x2, y2].
        confidence: Latest detection confidence.
        state: Current TrackState.
        hits: Total number of successful associations.
        age: Total number of frames since creation.
        time_since_update: Frames since last successful association.
        global_id: Global person ID from ReID gallery (None until assigned).
        frames_since_reid: Frames since last ReID feature extraction.
    """

    _next_id = 1

    def __init__(self, bbox, confidence):
        self.track_id = Track._next_id
        Track._next_id += 1

        self.bbox = list(bbox)
        self.confidence = confidence
        self.state = TrackState.TENTATIVE
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        # ReID-related
        self.global_id = None
        self.frames_since_reid = config.REID_UPDATE_INTERVAL  # Force ReID on first frame

        # Velocity estimation for simple prediction
        self._prev_center = self._center()

    def _center(self):
        """Get bbox center."""
        return np.array([
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        ])

    def predict(self):
        """
        Predict the next position using constant-velocity model.
        Called at the start of each frame before association.
        """
        center = self._center()
        velocity = center - self._prev_center

        # Shift bbox by velocity
        self.bbox[0] += velocity[0]
        self.bbox[1] += velocity[1]
        self.bbox[2] += velocity[0]
        self.bbox[3] += velocity[1]

        self.age += 1
        self.time_since_update += 1
        self.frames_since_reid += 1

    def update(self, bbox, confidence):
        """
        Update track with a matched detection.

        Args:
            bbox: New bounding box [x1, y1, x2, y2].
            confidence: Detection confidence.
        """
        self._prev_center = self._center()
        self.bbox = list(bbox)
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0

        # Transition from tentative to confirmed
        if self.state == TrackState.TENTATIVE and self.hits >= config.TRACKER_MIN_HITS:
            self.state = TrackState.CONFIRMED

    def mark_deleted(self):
        """Mark track as deleted."""
        self.state = TrackState.DELETED

    def is_confirmed(self):
        """Check if track is confirmed."""
        return self.state == TrackState.CONFIRMED

    def is_deleted(self):
        """Check if track is deleted."""
        return self.state == TrackState.DELETED

    def is_tentative(self):
        """Check if track is tentative."""
        return self.state == TrackState.TENTATIVE

    def needs_reid(self):
        """
        Check if this track needs ReID feature extraction.

        Returns True if:
        - Track is confirmed AND
        - (No global_id yet OR enough frames have passed since last update)
        """
        if not self.is_confirmed():
            return False
        if self.global_id is None:
            return True
        return self.frames_since_reid >= config.REID_UPDATE_INTERVAL

    def __repr__(self):
        return (
            f"Track(id={self.track_id}, global={self.global_id}, "
            f"state={self.state.name}, hits={self.hits}, "
            f"lost={self.time_since_update})"
        )


class IoUTracker:
    """
    Multi-object tracker using IoU-based association.

    Usage:
        tracker = IoUTracker()
        tracks = tracker.update(detections)  # Call per frame
    """

    def __init__(self):
        self.tracks = []
        self._frame_count = 0

    def update(self, detections):
        """
        Update tracker with new detections for the current frame.

        Args:
            detections: List of Detection objects with .bbox and .confidence.

        Returns:
            List of confirmed Track objects (active, visible tracks).
        """
        self._frame_count += 1

        # Step 1: Predict new positions for all existing tracks
        for track in self.tracks:
            track.predict()

        # Step 2: Associate detections with existing tracks
        if len(self.tracks) > 0 and len(detections) > 0:
            matched, unmatched_tracks, unmatched_dets = self._associate(
                self.tracks, detections
            )
        elif len(self.tracks) > 0:
            matched = []
            unmatched_tracks = list(range(len(self.tracks)))
            unmatched_dets = []
        else:
            matched = []
            unmatched_tracks = []
            unmatched_dets = list(range(len(detections)))

        # Step 3: Update matched tracks
        for track_idx, det_idx in matched:
            self.tracks[track_idx].update(
                detections[det_idx].bbox,
                detections[det_idx].confidence,
            )

        # Step 4: Handle unmatched tracks (lost)
        for track_idx in unmatched_tracks:
            track = self.tracks[track_idx]
            if track.time_since_update > config.TRACKER_MAX_AGE:
                track.mark_deleted()

        # Step 5: Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            new_track = Track(det.bbox, det.confidence)
            self.tracks.append(new_track)

        # Step 6: Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Return confirmed tracks
        confirmed = [t for t in self.tracks if t.is_confirmed()]
        logger.debug(
            f"Frame {self._frame_count}: "
            f"{len(confirmed)} confirmed, {len(self.tracks)} total tracks"
        )
        return confirmed

    def _associate(self, tracks, detections):
        """
        Associate existing tracks with new detections using IoU.

        Args:
            tracks: List of Track objects.
            detections: List of Detection objects.

        Returns:
            matched: List of (track_idx, det_idx) pairs.
            unmatched_tracks: List of unmatched track indices.
            unmatched_dets: List of unmatched detection indices.
        """
        # Build IoU cost matrix
        track_bboxes = np.array([t.bbox for t in tracks])
        det_bboxes = np.array([d.bbox for d in detections])
        iou_matrix = compute_iou_matrix(track_bboxes, det_bboxes)

        # Use Hungarian algorithm to find optimal assignment
        # scipy minimizes cost, so we use (1 - IoU) as cost
        cost_matrix = 1.0 - iou_matrix

        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        else:
            row_indices, col_indices = np.array([]), np.array([])

        # Filter assignments by IoU threshold
        matched = []
        unmatched_tracks = set(range(len(tracks)))
        unmatched_dets = set(range(len(detections)))

        for row, col in zip(row_indices, col_indices):
            if iou_matrix[row, col] >= config.TRACKER_IOU_THRESHOLD:
                matched.append((row, col))
                unmatched_tracks.discard(row)
                unmatched_dets.discard(col)

        return matched, list(unmatched_tracks), list(unmatched_dets)

    def get_all_tracks(self):
        """Get all tracks (including tentative)."""
        return self.tracks

    def reset(self):
        """Reset all tracks."""
        self.tracks = []
        self._frame_count = 0
        Track._next_id = 1
