"""
Hailo YOLOv8s person detector.

Uses the HailoRT Python API to run YOLOv8s inference on the Hailo-8L accelerator.
Parses raw output tensors into person detections with bounding boxes and confidence scores.
"""

import logging
import numpy as np

import config
from utils import non_max_suppression

logger = logging.getLogger(__name__)


class Detection:
    """A single person detection."""

    __slots__ = ["bbox", "confidence"]

    def __init__(self, bbox, confidence):
        """
        Args:
            bbox: [x1, y1, x2, y2] in pixel coordinates (relative to original frame).
            confidence: Detection confidence score.
        """
        self.bbox = bbox
        self.confidence = confidence

    def __repr__(self):
        return (
            f"Detection(bbox=[{self.bbox[0]:.0f},{self.bbox[1]:.0f},"
            f"{self.bbox[2]:.0f},{self.bbox[3]:.0f}], "
            f"conf={self.confidence:.2f})"
        )


class HailoDetector:
    """
    Person detector using YOLOv8s on the Hailo-8L accelerator.

    Lifecycle:
        detector = HailoDetector()
        detections = detector.detect(frame)
        detector.close()
    """

    def __init__(self, hef_path=None):
        """
        Initialize the Hailo device and load the YOLOv8s model.

        Args:
            hef_path: Path to the .hef model file.
                      Defaults to config.HEF_PATH.
        """
        from hailo_platform import VDevice, HEF, FormatType

        self.hef_path = hef_path or config.HEF_PATH
        logger.info(f"Loading Hailo model from: {self.hef_path}")

        # Initialize the virtual device
        self.vdevice = VDevice()

        # Load the HEF model
        self.hef = HEF(self.hef_path)

        # Create the inference model
        self.infer_model = self.vdevice.create_infer_model(self.hef_path)

        # Configure the model (allocates hardware resources)
        self.configured_model = self.infer_model.configure()

        # Cache input/output metadata
        self._input_vstream_info = self.hef.get_input_vstream_infos()
        self._output_vstream_info = self.hef.get_output_vstream_infos()

        # Get model input shape
        input_info = self._input_vstream_info[0]
        self.input_shape = input_info.shape  # e.g., (640, 640, 3)
        self.input_height = self.input_shape[0]
        self.input_width = self.input_shape[1]

        logger.info(
            f"Hailo model loaded. Input shape: {self.input_shape}, "
            f"Outputs: {len(self._output_vstream_info)}"
        )

        # Log output layer details
        for out_info in self._output_vstream_info:
            logger.debug(f"  Output layer: {out_info.name}, shape: {out_info.shape}")

    def preprocess(self, frame):
        """
        Preprocess frame for YOLOv8 inference.

        Args:
            frame: BGR frame from camera (H, W, 3).

        Returns:
            Preprocessed input tensor (model_H, model_W, 3) as uint8.
            Scale factors (scale_x, scale_y) for mapping detections back to original frame.
        """
        import cv2

        orig_h, orig_w = frame.shape[:2]

        # Resize to model input dimensions
        resized = cv2.resize(
            frame,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )

        # YOLOv8 HEF on Hailo typically expects uint8 BGR or RGB
        # The Hailo model handles quantization internally
        # Convert BGR to RGB if the model expects RGB
        input_tensor = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Scale factors for mapping detections back to original frame
        scale_x = orig_w / self.input_width
        scale_y = orig_h / self.input_height

        return input_tensor, (scale_x, scale_y)

    def detect(self, frame):
        """
        Run person detection on a frame.

        Args:
            frame: BGR frame from camera (H, W, 3).

        Returns:
            List of Detection objects (persons only, after NMS).
        """
        # Preprocess
        input_tensor, (scale_x, scale_y) = self.preprocess(frame)

        # Create bindings and set input
        bindings = self.configured_model.create_bindings()
        input_name = self._input_vstream_info[0].name
        bindings.input(input_name).set_buffer(
            np.expand_dims(input_tensor, axis=0)
        )

        # Run synchronous inference
        self.configured_model.wait_for_async_ready(timeout_ms=10000)
        job = self.configured_model.run_async(bindings)
        job.wait(timeout_ms=10000)

        # Extract raw outputs
        raw_outputs = {}
        for out_info in self._output_vstream_info:
            raw_outputs[out_info.name] = bindings.output(out_info.name).get_buffer()

        # Parse detections from raw output
        detections = self._parse_yolov8_output(raw_outputs, scale_x, scale_y)

        # Apply NMS
        detections = non_max_suppression(detections, config.NMS_IOU_THRESHOLD)

        logger.debug(f"Detected {len(detections)} persons")
        return detections

    def _parse_yolov8_output(self, raw_outputs, scale_x, scale_y):
        """
        Parse YOLOv8 output tensors into Detection objects.

        Hailo YOLOv8 outputs are typically split across multiple output layers
        corresponding to different detection heads (strides 8, 16, 32).
        Each output contains bounding box regression + class probabilities.

        Args:
            raw_outputs: Dict of output_name -> np.ndarray.
            scale_x: X-axis scale factor to map to original frame.
            scale_y: Y-axis scale factor to map to original frame.

        Returns:
            List of Detection objects (person class only).
        """
        detections = []

        for layer_name, output in raw_outputs.items():
            # Remove batch dimension if present
            if output.ndim == 4:
                output = output[0]

            # Hailo YOLOv8 output format: (H, W, num_detections_per_cell)
            # where the channels contain [x, y, w, h, class_scores...]
            # or in some configurations: (H, W, (4 + num_classes))
            h, w, channels = output.shape

            # Determine format based on channel count
            if channels == 4 + config.NUM_CLASSES:
                # Standard format: [x, y, w, h, class_0, class_1, ...]
                bbox_data = output[:, :, :4]
                class_scores = output[:, :, 4:]
            elif channels % (4 + config.NUM_CLASSES) == 0:
                # Multi-anchor format
                anchors_per_cell = channels // (4 + config.NUM_CLASSES)
                output = output.reshape(h, w, anchors_per_cell, 4 + config.NUM_CLASSES)
                for a in range(anchors_per_cell):
                    self._extract_detections_from_grid(
                        output[:, :, a, :4],
                        output[:, :, a, 4:],
                        h, w, scale_x, scale_y, detections,
                    )
                continue
            else:
                # Try parsing as a flat detection list
                # Some Hailo outputs come as (1, N, 4+C) or (N, 4+C)
                if output.ndim == 2 or (output.ndim == 3 and output.shape[0] == 1):
                    flat = output.reshape(-1, output.shape[-1])
                    if flat.shape[1] >= 5:
                        self._extract_detections_flat(
                            flat, scale_x, scale_y, detections
                        )
                continue

            self._extract_detections_from_grid(
                bbox_data, class_scores, h, w, scale_x, scale_y, detections
            )

        return detections

    def _extract_detections_from_grid(
        self, bbox_data, class_scores, grid_h, grid_w,
        scale_x, scale_y, detections
    ):
        """
        Extract person detections from a grid-format output.

        Args:
            bbox_data: (H, W, 4) — [cx, cy, w, h] normalized or strided.
            class_scores: (H, W, num_classes) — class probabilities.
            grid_h, grid_w: Grid dimensions.
            scale_x, scale_y: Scale factors to original frame.
            detections: List to append Detection objects to.
        """
        stride_x = self.input_width / grid_w
        stride_y = self.input_height / grid_h

        # Get person class scores
        person_scores = class_scores[:, :, config.PERSON_CLASS_ID]

        # Find cells where person confidence exceeds threshold
        mask = person_scores > config.DETECTION_CONFIDENCE
        ys, xs = np.where(mask)

        for y, x in zip(ys, xs):
            confidence = float(person_scores[y, x])
            cx, cy, bw, bh = bbox_data[y, x]

            # Convert from grid-relative to pixel coordinates
            # YOLOv8 outputs center-x, center-y, width, height
            cx = (float(cx) + x) * stride_x
            cy = (float(cy) + y) * stride_y
            bw = float(bw) * stride_x
            bh = float(bh) * stride_y

            # Convert to [x1, y1, x2, y2] in original frame coordinates
            x1 = (cx - bw / 2) * scale_x
            y1 = (cy - bh / 2) * scale_y
            x2 = (cx + bw / 2) * scale_x
            y2 = (cy + bh / 2) * scale_y

            detections.append(Detection(
                bbox=[x1, y1, x2, y2],
                confidence=confidence,
            ))

    def _extract_detections_flat(self, flat_output, scale_x, scale_y, detections):
        """
        Extract detections from a flat (N, 4+C) or (N, 5+) output format.

        Some Hailo post-processing configurations output a flat list of detections
        with format [x1, y1, x2, y2, confidence, class_id] or similar.

        Args:
            flat_output: (N, D) array of detections.
            scale_x, scale_y: Scale factors.
            detections: List to append to.
        """
        num_cols = flat_output.shape[1]

        for row in flat_output:
            if num_cols == 6:
                # [x1, y1, x2, y2, confidence, class_id]
                x1, y1, x2, y2, conf, cls_id = row
                cls_id = int(cls_id)
            elif num_cols == 5:
                # [x1, y1, x2, y2, confidence] — already filtered
                x1, y1, x2, y2, conf = row
                cls_id = config.PERSON_CLASS_ID
            elif num_cols >= 4 + config.NUM_CLASSES:
                # [x1, y1, x2, y2, class_0, class_1, ...]
                x1, y1, x2, y2 = row[:4]
                cls_scores = row[4:]
                cls_id = int(np.argmax(cls_scores))
                conf = float(cls_scores[cls_id])
            else:
                continue

            if cls_id != config.PERSON_CLASS_ID:
                continue
            if conf < config.DETECTION_CONFIDENCE:
                continue

            # Scale to original frame
            detections.append(Detection(
                bbox=[
                    float(x1) * scale_x,
                    float(y1) * scale_y,
                    float(x2) * scale_x,
                    float(y2) * scale_y,
                ],
                confidence=float(conf),
            ))

    def close(self):
        """Release Hailo resources."""
        logger.info("Releasing Hailo resources")
        try:
            del self.configured_model
            del self.infer_model
            del self.vdevice
        except Exception as e:
            logger.warning(f"Error releasing Hailo resources: {e}")

    def __del__(self):
        self.close()
