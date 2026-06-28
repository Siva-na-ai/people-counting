# People Counter — Raspberry Pi 5 + Hailo AI HAT

Real-time unique people counting system with person re-identification. Detects persons using YOLOv8s on the Hailo-8L accelerator, tracks them frame-to-frame, and uses OSNet to recognize individuals — even after they leave and re-enter the scene.

**All data is stored in RAM** — no database required.

## Features

- **Hardware-Accelerated Detection** — YOLOv8s runs on the Hailo-8L AI accelerator (~30 FPS)
- **Multi-Object Tracking** — IoU-based tracker with Hungarian assignment for stable frame-to-frame tracking
- **Person Re-Identification** — OSNet extracts 512-dim appearance embeddings for each person
- **Global ID Gallery** — In-memory gallery matches new detections against known persons using cosine similarity
- **Re-Entry Detection** — Recognizes the same person when they leave and come back
- **EMA Embedding Smoothing** — Exponential moving average keeps embeddings stable over time
- **Live HUD** — On-screen display with unique count, active count, re-entries, and FPS
- **Flexible Input** — PiCamera2, USB webcam, video file, or RTSP stream
- **Headless Mode** — Run without display for embedded deployments

## Architecture

```
Camera → YOLOv8s (Hailo) → IoU Tracker → OSNet ReID → Gallery → Unique Count
```

| Component | Module | Description |
|-----------|--------|-------------|
| Detection | `detector.py` | Hailo HEF inference with YOLOv8 output parsing |
| Tracking | `tracker.py` | SORT-inspired IoU tracker with Hungarian algorithm |
| ReID | `reid.py` | OSNet feature extraction with ImageNet normalization |
| Gallery | `gallery.py` | In-memory gallery with EMA embedding updates |
| Visualization | `visualizer.py` | Bounding boxes, labels, and HUD overlay |
| Config | `config.py` | All tunable parameters in one file |
| Main | `people_counter.py` | Pipeline orchestrator with CLI interface |

## Requirements

### Hardware
- Raspberry Pi 5 (4GB+ RAM recommended)
- Hailo-8L AI HAT (M.2 or HAT+ form factor)
- PiCamera2 module or USB webcam

### Software
- Raspberry Pi OS (64-bit, Bookworm or later)
- HailoRT SDK installed (provides `hailo_platform`)
- Python 3.11+

## Quick Start

### 1. Clone & Setup

```bash
git clone <your-repo-url> people-counting
cd people-counting
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Install system dependencies
- Create a Python virtual environment
- Install all Python packages
- Clone and install `torchreid` (for OSNet)
- Download pretrained OSNet weights
- Verify Hailo device connectivity

### 2. Activate Environment

```bash
source venv/bin/activate
```

### 3. Run

```bash
# Default: PiCamera2
python people_counter.py

# With a video file
python people_counter.py --source /path/to/video.mp4

# USB webcam
python people_counter.py --source usb:0

# Headless mode (no display)
python people_counter.py --no-display

# Save annotated output video
python people_counter.py --save-video output.avi

# Custom thresholds
python people_counter.py --confidence 0.6 --match-threshold 0.5

# Debug mode (verbose logging)
python people_counter.py --log-level DEBUG
```

### 4. Controls (when display is enabled)

| Key | Action |
|-----|--------|
| `q` or `ESC` | Quit |
| `r` | Reset gallery and tracker |

## Configuration

All parameters are in [`config.py`](config.py). Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HEF_PATH` | `/usr/share/hailo-models/yolov8s_h8l.hef` | Hailo model path |
| `DETECTION_CONFIDENCE` | `0.5` | Minimum detection confidence |
| `GALLERY_MATCH_THRESHOLD` | `0.55` | Cosine similarity for re-identification |
| `GALLERY_EMA_ALPHA` | `0.9` | Embedding smoothing weight |
| `TRACKER_MAX_AGE` | `30` | Frames before losing a track |
| `REID_UPDATE_INTERVAL` | `15` | Frames between ReID updates |

### Environment Variables

```bash
export HEF_PATH="/custom/path/to/model.hef"
export INPUT_SOURCE="usb:0"
```

## How It Works

### Pipeline (per frame)

1. **Capture** — Read frame from PiCamera2 / webcam / video
2. **Detect** — YOLOv8s on Hailo → person bounding boxes
3. **Track** — IoU matching + Hungarian assignment → stable track IDs
4. **ReID** — OSNet extracts 512-dim embedding for new/updated tracks
5. **Gallery Match** — Cosine similarity against known persons:
   - **Match found (≥ 0.55)** → Assign existing global ID, update embedding via EMA
   - **No match** → Create new global ID
6. **Visualize** — Draw boxes, labels, HUD with live statistics

### Re-Entry Detection

When a person leaves the frame, their track is deleted but their embedding stays in the gallery. When they return:
1. A new track is created by the tracker
2. ReID extracts their appearance embedding
3. Gallery matching finds the closest match
4. If similarity ≥ threshold → same global ID is reassigned
5. Re-entry counter increments

## Folder Structure

```
people-counting/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── setup.sh                  # One-command setup for RPi5
├── config.py                 # All tunable parameters
├── people_counter.py         # Main entry point
├── detector.py               # Hailo YOLOv8 detection
├── tracker.py                # IoU multi-object tracker
├── reid.py                   # OSNet ReID extraction
├── gallery.py                # In-memory person gallery
├── utils.py                  # Bbox/NMS/similarity utilities
├── visualizer.py             # Drawing & HUD overlay
└── third_party/              # (created by setup.sh)
    └── deep-person-reid/     # torchreid source
```

## Troubleshooting

### Hailo device not detected
```bash
# Check if Hailo is visible
lspci | grep Hailo
# Check HailoRT service
hailortcli fw-control identify
```

### Low FPS
- Increase `REID_UPDATE_INTERVAL` in `config.py` (default: 15)
- Lower camera resolution: set `CAMERA_WIDTH=640`, `CAMERA_HEIGHT=480`
- Use `--no-display` flag for headless operation

### False re-identifications (wrong matches)
- Increase `GALLERY_MATCH_THRESHOLD` (e.g., from 0.55 to 0.65)
- This makes matching stricter but may miss some real re-entries

### Missed re-entries
- Decrease `GALLERY_MATCH_THRESHOLD` (e.g., from 0.55 to 0.45)
- Note: lower threshold increases false positive risk

### torchreid import error
```bash
cd third_party/deep-person-reid
pip install -e .
```

## License

MIT
