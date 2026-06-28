"""
Utility functions for bounding box operations, NMS, and image preprocessing.
"""

import numpy as np


def compute_iou(bbox1, bbox2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.

    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]

    Returns:
        IoU value (float between 0 and 1).
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0
    return intersection / union


def compute_iou_matrix(bboxes1, bboxes2):
    """
    Compute IoU matrix between two sets of bounding boxes.

    Args:
        bboxes1: np.ndarray of shape (N, 4), format [x1, y1, x2, y2]
        bboxes2: np.ndarray of shape (M, 4), format [x1, y1, x2, y2]

    Returns:
        IoU matrix of shape (N, M).
    """
    n = len(bboxes1)
    m = len(bboxes2)
    iou_matrix = np.zeros((n, m), dtype=np.float32)

    for i in range(n):
        for j in range(m):
            iou_matrix[i, j] = compute_iou(bboxes1[i], bboxes2[j])

    return iou_matrix


def non_max_suppression(detections, iou_threshold):
    """
    Apply Non-Maximum Suppression to filter overlapping detections.

    Args:
        detections: List of Detection objects with .bbox and .confidence attributes.
        iou_threshold: IoU threshold above which the lower-confidence detection is suppressed.

    Returns:
        Filtered list of Detection objects.
    """
    if len(detections) == 0:
        return []

    # Sort by confidence descending
    sorted_dets = sorted(detections, key=lambda d: d.confidence, reverse=True)
    keep = []

    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)

        remaining = []
        for det in sorted_dets:
            if compute_iou(best.bbox, det.bbox) < iou_threshold:
                remaining.append(det)
        sorted_dets = remaining

    return keep


def clip_bbox(bbox, frame_width, frame_height):
    """
    Clip bounding box coordinates to frame boundaries.

    Args:
        bbox: [x1, y1, x2, y2]
        frame_width: Width of the frame.
        frame_height: Height of the frame.

    Returns:
        Clipped [x1, y1, x2, y2].
    """
    x1 = max(0, min(int(bbox[0]), frame_width - 1))
    y1 = max(0, min(int(bbox[1]), frame_height - 1))
    x2 = max(0, min(int(bbox[2]), frame_width - 1))
    y2 = max(0, min(int(bbox[3]), frame_height - 1))
    return [x1, y1, x2, y2]


def crop_person(frame, bbox, target_size=None):
    """
    Crop a person from the frame using their bounding box, with optional resize.

    Args:
        frame: Full frame (H, W, 3) BGR.
        bbox: [x1, y1, x2, y2] in pixel coordinates.
        target_size: Optional (height, width) to resize the crop with letterboxing.

    Returns:
        Cropped (and optionally resized) image as np.ndarray.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = clip_bbox(bbox, w, h)

    # Ensure valid crop
    if x2 <= x1 or y2 <= y1:
        # Return a blank crop of target size or minimum size
        if target_size:
            return np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        return np.zeros((128, 64, 3), dtype=np.uint8)

    crop = frame[y1:y2, x1:x2].copy()

    if target_size is not None:
        crop = letterbox_resize(crop, target_size)

    return crop


def letterbox_resize(image, target_size):
    """
    Resize image to target size while preserving aspect ratio with letterboxing.

    Args:
        image: Input image (H, W, 3).
        target_size: (target_height, target_width).

    Returns:
        Resized and padded image.
    """
    import cv2

    th, tw = target_size
    h, w = image.shape[:2]

    # Compute scale to fit within target
    scale = min(tw / w, th / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Create padded canvas
    canvas = np.full((th, tw, 3), 128, dtype=np.uint8)
    offset_x = (tw - new_w) // 2
    offset_y = (th - new_h) // 2
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

    return canvas


def bbox_area(bbox):
    """Compute area of a bounding box [x1, y1, x2, y2]."""
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def bbox_center(bbox):
    """Compute center (cx, cy) of a bounding box [x1, y1, x2, y2]."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def cosine_similarity(feat1, feat2):
    """
    Compute cosine similarity between two L2-normalized feature vectors.

    Args:
        feat1: np.ndarray of shape (D,)
        feat2: np.ndarray of shape (D,)

    Returns:
        Cosine similarity (float, range [-1, 1]).
    """
    return float(np.dot(feat1, feat2))


def cosine_similarity_matrix(feats1, feats2):
    """
    Compute cosine similarity matrix between two sets of L2-normalized features.

    Args:
        feats1: np.ndarray of shape (N, D)
        feats2: np.ndarray of shape (M, D)

    Returns:
        Similarity matrix of shape (N, M).
    """
    return np.dot(feats1, feats2.T)
