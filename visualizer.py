"""
Visualization module for the People Counting System.

Draws annotated frames with bounding boxes, global IDs,
confidence scores, and a heads-up display (HUD) with live statistics.
"""

import cv2
import numpy as np

import config


def get_id_color(global_id):
    """
    Get a consistent color for a given global ID.

    Args:
        global_id: Person's global ID.

    Returns:
        BGR color tuple.
    """
    if global_id is None:
        return (128, 128, 128)  # Gray for unidentified
    idx = (global_id - 1) % len(config.ID_COLORS)
    return config.ID_COLORS[idx]


def draw_tracks(frame, tracks, gallery=None):
    """
    Draw bounding boxes and labels for all tracked persons.

    Args:
        frame: BGR frame to draw on (modified in-place).
        tracks: List of Track objects.
        gallery: Optional Gallery instance for person info.

    Returns:
        Annotated frame.
    """
    for track in tracks:
        bbox = track.bbox
        global_id = track.global_id
        color = get_id_color(global_id)

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Build label text
        if global_id is not None:
            label = f"ID:{global_id}"
        else:
            label = f"T:{track.track_id}"

        label += f" {track.confidence:.0%}"

        # Check for re-entry indicator
        if gallery and global_id is not None:
            person = gallery.get_person(global_id)
            if person and person.re_entries > 0:
                label += f" R:{person.re_entries}"

        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = config.HUD_FONT_SCALE
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, thickness
        )

        # Label above the box
        label_y = max(y1 - 10, text_h + 5)
        cv2.rectangle(
            frame,
            (x1, label_y - text_h - 5),
            (x1 + text_w + 8, label_y + 3),
            color,
            -1,
        )

        # Choose text color for readability
        brightness = sum(color) / 3
        text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)

        cv2.putText(
            frame, label, (x1 + 4, label_y - 2),
            font, font_scale, text_color, thickness, cv2.LINE_AA,
        )

        # Draw a small filled circle at bbox center (tracking dot)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.circle(frame, (cx, cy), 4, color, -1)

    return frame


def draw_hud(frame, stats, fps=0.0):
    """
    Draw a heads-up display (HUD) overlay with system statistics.

    Args:
        frame: BGR frame to draw on.
        stats: Dict from Gallery.get_stats() with keys:
               unique_count, active_count, re_entries, gallery_size.
        fps: Current frames per second.

    Returns:
        Annotated frame.
    """
    h, w = frame.shape[:2]

    # HUD dimensions
    hud_w = 280
    hud_h = 160
    margin = 10
    hud_x = w - hud_w - margin
    hud_y = margin

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (hud_x, hud_y),
        (hud_x + hud_w, hud_y + hud_h),
        (20, 20, 20),
        -1,
    )
    cv2.addWeighted(
        overlay, config.HUD_BG_ALPHA,
        frame, 1 - config.HUD_BG_ALPHA,
        0, frame,
    )

    # Border
    cv2.rectangle(
        frame,
        (hud_x, hud_y),
        (hud_x + hud_w, hud_y + hud_h),
        (0, 255, 200),
        1,
    )

    # Title bar
    cv2.rectangle(
        frame,
        (hud_x, hud_y),
        (hud_x + hud_w, hud_y + 28),
        (0, 255, 200),
        -1,
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    bold_font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    cv2.putText(
        frame, "PEOPLE COUNTER",
        (hud_x + 10, hud_y + 20),
        bold_font, 0.55, (0, 0, 0), 2, cv2.LINE_AA,
    )

    # Stats lines
    text_x = hud_x + 15
    line_height = 26
    start_y = hud_y + 52
    font_scale = 0.50
    thickness = 1

    lines = [
        (f"Unique People:   {stats.get('unique_count', 0)}", (0, 255, 200)),
        (f"Currently Visible: {stats.get('active_count', 0)}", (255, 255, 255)),
        (f"Re-entries:       {stats.get('re_entries', 0)}", (100, 200, 255)),
        (f"FPS:              {fps:.1f}", (200, 200, 200)),
    ]

    for i, (text, color) in enumerate(lines):
        y = start_y + i * line_height
        cv2.putText(
            frame, text, (text_x, y),
            font, font_scale, color, thickness, cv2.LINE_AA,
        )

    return frame


def draw_frame(frame, tracks, stats, fps=0.0, gallery=None):
    """
    Full frame annotation: tracks + HUD.

    Args:
        frame: BGR frame.
        tracks: List of Track objects.
        stats: Gallery stats dict.
        fps: Current FPS.
        gallery: Optional Gallery instance.

    Returns:
        Fully annotated frame.
    """
    annotated = frame.copy()
    annotated = draw_tracks(annotated, tracks, gallery)
    annotated = draw_hud(annotated, stats, fps)
    return annotated
