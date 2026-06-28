"""
Configuration constants for the People Counting System.
All tunable parameters are centralized here.
"""

import os

# ==============================================================================
# Hailo Detection Settings
# ==============================================================================

# Path to the Hailo YOLOv8s model file
HEF_PATH = os.environ.get(
    "HEF_PATH",
    "/usr/share/hailo-models/yolov8s_h8l.hef"
)

# Minimum confidence threshold for person detections
DETECTION_CONFIDENCE = 0.5

# Non-Maximum Suppression IoU threshold
NMS_IOU_THRESHOLD = 0.45

# COCO class ID for "person"
PERSON_CLASS_ID = 0

# YOLOv8s input dimensions (must match the HEF model)
MODEL_INPUT_WIDTH = 640
MODEL_INPUT_HEIGHT = 640

# Number of COCO classes
NUM_CLASSES = 80

# ==============================================================================
# Tracker Settings
# ==============================================================================

# Maximum frames a track can be "lost" before deletion
TRACKER_MAX_AGE = 30

# Minimum consecutive hits before a track is confirmed
TRACKER_MIN_HITS = 3

# Minimum IoU to associate a detection with an existing track
TRACKER_IOU_THRESHOLD = 0.3

# ==============================================================================
# ReID (OSNet) Settings
# ==============================================================================

# OSNet model variant
REID_MODEL_NAME = "osnet_x1_0"

# ReID crop input size (Height x Width) - standard for person ReID
REID_INPUT_HEIGHT = 256
REID_INPUT_WIDTH = 128

# Device for ReID inference ("cpu" for RPi5, "cuda" if GPU available)
REID_DEVICE = "cpu"

# Frames between ReID embedding updates for active tracks
REID_UPDATE_INTERVAL = 15

# ==============================================================================
# Gallery Settings
# ==============================================================================

# Cosine similarity threshold for matching a person to the gallery.
# Above this → same person. Below this → new person.
# Tuned conservatively to reduce false merges; lower if you're missing re-entries.
GALLERY_MATCH_THRESHOLD = 0.55

# Exponential Moving Average weight for embedding smoothing.
# Higher = more weight to old embedding (more stable, slower to adapt).
GALLERY_EMA_ALPHA = 0.9

# Maximum gallery size before pruning oldest entries (0 = unlimited)
GALLERY_MAX_SIZE = 0

# ==============================================================================
# Camera / Input Settings
# ==============================================================================

# Input source: "picamera" | "usb:0" | "/path/to/video.mp4" | "rtsp://..."
INPUT_SOURCE = os.environ.get("INPUT_SOURCE", "picamera")

# Camera resolution
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Target FPS (for PiCamera2 configuration)
CAMERA_FPS = 30

# ==============================================================================
# Visualization Settings
# ==============================================================================

# Show live visualization window
DISPLAY_ENABLED = True

# HUD overlay settings
HUD_FONT_SCALE = 0.6
HUD_THICKNESS = 2
HUD_BG_ALPHA = 0.7

# Color palette for global IDs (BGR format for OpenCV)
# 20 distinct colors — wraps around for IDs > 20
ID_COLORS = [
    (255, 85, 85),    # Red
    (85, 255, 85),    # Green
    (85, 85, 255),    # Blue
    (255, 255, 85),   # Yellow
    (255, 85, 255),   # Magenta
    (85, 255, 255),   # Cyan
    (255, 170, 85),   # Orange
    (170, 85, 255),   # Purple
    (85, 255, 170),   # Spring Green
    (255, 85, 170),   # Rose
    (170, 255, 85),   # Lime
    (85, 170, 255),   # Sky Blue
    (255, 170, 170),  # Light Coral
    (170, 255, 170),  # Light Green
    (170, 170, 255),  # Light Blue
    (255, 255, 170),  # Light Yellow
    (255, 170, 255),  # Light Magenta
    (170, 255, 255),  # Light Cyan
    (200, 200, 200),  # Silver
    (128, 128, 255),  # Periwinkle
]

# ==============================================================================
# Logging Settings
# ==============================================================================

# Seconds between console stat prints
LOG_INTERVAL = 5

# Log level: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL = "INFO"
