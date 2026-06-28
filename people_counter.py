#!/usr/bin/env python3
"""
People Counter — Main Orchestrator

Real-time unique people counting system for Raspberry Pi 5 + Hailo AI HAT.

Pipeline:
    Camera → YOLOv8s (Hailo) → IoU Tracker → OSNet ReID → Gallery → Count

Usage:
    python people_counter.py                          # PiCamera2 (default)
    python people_counter.py --source /path/to.mp4    # Video file
    python people_counter.py --source usb:0           # USB webcam
    python people_counter.py --no-display              # Headless mode
    python people_counter.py --save-video output.avi   # Save annotated video

Author: Re-Shield
"""

import argparse
import logging
import signal
import sys
import time

import cv2
import numpy as np

import config
from detector import HailoDetector
from tracker import IoUTracker
from reid import ReIDExtractor
from gallery import Gallery
from visualizer import draw_frame


# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(level):
    """Configure logging format and level."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Camera / Video Input
# ─────────────────────────────────────────────────────────────────────────────

class CameraInput:
    """Unified camera/video input handler."""

    def __init__(self, source):
        """
        Initialize the input source.

        Args:
            source: "picamera" | "usb:N" | file path | RTSP URL
        """
        self.source = source
        self.cap = None
        self._picam2 = None
        self.logger = logging.getLogger("CameraInput")

        if source == "picamera":
            self._init_picamera()
        elif source.startswith("usb:"):
            device_id = int(source.split(":")[1])
            self._init_opencv(device_id)
        elif source.startswith("rtsp://") or source.startswith("http://"):
            self._init_opencv(source)
        else:
            # Assume file path
            self._init_opencv(source)

    def _init_picamera(self):
        """Initialize PiCamera2."""
        try:
            from picamera2 import Picamera2

            self._picam2 = Picamera2()
            cam_config = self._picam2.create_preview_configuration(
                main={"size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
                      "format": "RGB888"},
            )
            self._picam2.configure(cam_config)
            self._picam2.start()
            self.logger.info(
                f"PiCamera2 started: {config.CAMERA_WIDTH}x{config.CAMERA_HEIGHT}"
            )
        except ImportError:
            self.logger.error("picamera2 not installed. Install with: pip install picamera2")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize PiCamera2: {e}")
            raise

    def _init_opencv(self, source):
        """Initialize OpenCV VideoCapture."""
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        # Set resolution for live cameras
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.logger.info(f"OpenCV source opened: {actual_w}x{actual_h}")

    def read(self):
        """
        Read a frame from the input source.

        Returns:
            BGR frame as np.ndarray, or None if stream ended.
        """
        if self._picam2 is not None:
            frame = self._picam2.capture_array()
            # PiCamera2 returns RGB, convert to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        elif self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                return None
            return frame
        return None

    def release(self):
        """Release camera resources."""
        if self._picam2 is not None:
            self._picam2.stop()
            self._picam2.close()
            self.logger.info("PiCamera2 released")
        if self.cap is not None:
            self.cap.release()
            self.logger.info("OpenCV capture released")


# ─────────────────────────────────────────────────────────────────────────────
# FPS Counter
# ─────────────────────────────────────────────────────────────────────────────

class FPSCounter:
    """Tracks and smooths FPS measurement."""

    def __init__(self, window=30):
        self._window = window
        self._timestamps = []

    def tick(self):
        """Record a frame timestamp."""
        now = time.time()
        self._timestamps.append(now)
        # Keep only last N timestamps
        if len(self._timestamps) > self._window:
            self._timestamps = self._timestamps[-self._window:]

    def fps(self):
        """Get current smoothed FPS."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed


# ─────────────────────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class PeopleCounter:
    """
    Main people counting pipeline.

    Integrates detection, tracking, ReID, gallery matching, and visualization
    into a single cohesive system.
    """

    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger("PeopleCounter")
        self.running = True

        # Register signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.logger.info("=" * 60)
        self.logger.info("  People Counter — Initializing")
        self.logger.info("=" * 60)

        # Initialize components
        self.logger.info("Initializing camera...")
        self.camera = CameraInput(args.source)

        self.logger.info("Initializing Hailo detector...")
        self.detector = HailoDetector(args.hef)

        self.logger.info("Initializing tracker...")
        self.tracker = IoUTracker()

        self.logger.info("Initializing ReID extractor...")
        self.reid = ReIDExtractor()

        self.logger.info("Initializing gallery...")
        self.gallery = Gallery()

        # FPS tracking
        self.fps_counter = FPSCounter()

        # Video writer
        self.video_writer = None
        if args.save_video:
            self._init_video_writer(args.save_video)

        # Stats logging
        self._last_log_time = time.time()

        self.logger.info("=" * 60)
        self.logger.info("  All components initialized. Starting pipeline.")
        self.logger.info("=" * 60)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}. Shutting down...")
        self.running = False

    def _init_video_writer(self, output_path):
        """Initialize video writer for saving annotated output."""
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.video_writer = cv2.VideoWriter(
            output_path,
            fourcc,
            config.CAMERA_FPS,
            (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
        )
        self.logger.info(f"Video writer initialized: {output_path}")

    def run(self):
        """Main processing loop."""
        frame_count = 0

        try:
            while self.running:
                # 1. Capture frame
                frame = self.camera.read()
                if frame is None:
                    self.logger.info("End of video stream")
                    break

                frame_count += 1
                self.fps_counter.tick()

                # 2. Detect persons (Hailo YOLOv8s)
                detections = self.detector.detect(frame)

                # 3. Track detections (IoU Tracker)
                tracks = self.tracker.update(detections)

                # 4. ReID + Gallery matching for tracks that need it
                # First, mark all gallery entries as inactive for this frame
                active_global_ids = set()

                for track in tracks:
                    if track.needs_reid():
                        # Extract ReID embedding
                        embedding = self.reid.extract_single(frame, track.bbox)

                        if np.linalg.norm(embedding) > 1e-6:
                            # Match against gallery
                            global_id, is_new = self.gallery.match_and_assign(
                                embedding, track.confidence
                            )

                            track.global_id = global_id
                            track.frames_since_reid = 0

                            # Update gallery embedding (EMA)
                            if not is_new:
                                self.gallery.update_embedding(global_id, embedding)

                    if track.global_id is not None:
                        active_global_ids.add(track.global_id)

                # Mark persons not currently tracked as inactive
                for gid, person in self.gallery.get_all_persons().items():
                    if gid in active_global_ids:
                        person.is_active = True
                    else:
                        person.is_active = False

                # 5. Get stats
                stats = self.gallery.get_stats()
                current_fps = self.fps_counter.fps()

                # 6. Visualize
                if self.args.display or self.video_writer is not None:
                    annotated = draw_frame(
                        frame, tracks, stats,
                        fps=current_fps,
                        gallery=self.gallery,
                    )

                    if self.args.display:
                        cv2.imshow("People Counter", annotated)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord("q") or key == 27:  # 'q' or ESC
                            self.logger.info("User requested quit")
                            break
                        elif key == ord("r"):
                            self.logger.info("Gallery reset by user")
                            self.gallery.reset()
                            self.tracker.reset()

                    if self.video_writer is not None:
                        self.video_writer.write(annotated)

                # 7. Periodic console logging
                now = time.time()
                if now - self._last_log_time >= self.args.log_interval:
                    self._log_stats(stats, current_fps, frame_count)
                    self._last_log_time = now

        except Exception as e:
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
            raise

        finally:
            self._cleanup()

    def _log_stats(self, stats, fps, frame_count):
        """Print statistics to console."""
        self.logger.info(
            f"[Frame {frame_count}] "
            f"FPS: {fps:.1f} | "
            f"Unique People: {stats['unique_count']} | "
            f"Visible: {stats['active_count']} | "
            f"Re-entries: {stats['re_entries']} | "
            f"Gallery Size: {stats['gallery_size']}"
        )

    def _cleanup(self):
        """Release all resources."""
        self.logger.info("Cleaning up resources...")

        if self.video_writer is not None:
            self.video_writer.release()
            self.logger.info("Video writer released")

        self.camera.release()
        self.detector.close()
        cv2.destroyAllWindows()

        # Final stats
        stats = self.gallery.get_stats()
        self.logger.info("=" * 60)
        self.logger.info("  FINAL RESULTS")
        self.logger.info(f"  Total Unique People:  {stats['unique_count']}")
        self.logger.info(f"  Total Re-entries:     {stats['re_entries']}")
        self.logger.info(f"  Gallery Size:         {stats['gallery_size']}")
        self.logger.info("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="People Counter — Hailo YOLOv8 + OSNet ReID",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--source",
        type=str,
        default=config.INPUT_SOURCE,
        help='Input source: "picamera", "usb:0", video file path, or RTSP URL',
    )
    parser.add_argument(
        "--hef",
        type=str,
        default=config.HEF_PATH,
        help="Path to Hailo HEF model file",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=config.DISPLAY_ENABLED,
        help="Show live visualization window",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        default=False,
        help="Disable visualization (headless mode)",
    )
    parser.add_argument(
        "--save-video",
        type=str,
        default=None,
        help="Path to save annotated output video (e.g., output.avi)",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=config.LOG_INTERVAL,
        help="Seconds between console stat prints",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=config.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=config.DETECTION_CONFIDENCE,
        help="Minimum detection confidence threshold",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=config.GALLERY_MATCH_THRESHOLD,
        help="Cosine similarity threshold for ReID matching",
    )

    args = parser.parse_args()

    # Handle --no-display flag
    if args.no_display:
        args.display = False

    # Apply CLI overrides to config
    config.DETECTION_CONFIDENCE = args.confidence
    config.GALLERY_MATCH_THRESHOLD = args.match_threshold

    return args


def main():
    """Entry point."""
    args = parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger("main")
    logger.info("People Counter starting...")
    logger.info(f"  Source:           {args.source}")
    logger.info(f"  HEF Model:       {args.hef}")
    logger.info(f"  Display:         {args.display}")
    logger.info(f"  Confidence:      {args.confidence}")
    logger.info(f"  Match Threshold: {args.match_threshold}")

    counter = PeopleCounter(args)
    counter.run()


if __name__ == "__main__":
    main()
