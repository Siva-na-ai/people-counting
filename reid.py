"""
OSNet-based person Re-Identification (ReID) feature extractor.

Extracts 512-dimensional embedding vectors from person crops using OSNet.
Features are L2-normalized for cosine similarity matching.
"""

import logging

import cv2
import numpy as np
import torch
import torchvision.transforms as T

import config

logger = logging.getLogger(__name__)


class ReIDExtractor:
    """
    Person Re-Identification feature extractor using OSNet.

    Extracts L2-normalized 512-dimensional embeddings from person image crops.
    Runs on CPU for Raspberry Pi 5 (no CUDA).

    Usage:
        extractor = ReIDExtractor()
        embeddings = extractor.extract(frame, [bbox1, bbox2, ...])
    """

    def __init__(self, model_name=None, device=None):
        """
        Initialize OSNet model with pretrained weights.

        Args:
            model_name: OSNet variant (default: config.REID_MODEL_NAME).
            device: Torch device (default: config.REID_DEVICE).
        """
        self.model_name = model_name or config.REID_MODEL_NAME
        self.device = torch.device(device or config.REID_DEVICE)

        logger.info(f"Loading ReID model: {self.model_name} on {self.device}")

        # Build OSNet model using torchreid
        try:
            import torchreid

            self.model = torchreid.models.build_model(
                name=self.model_name,
                num_classes=1000,  # Not used for feature extraction
                pretrained=True,
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            self._feature_dim = self._get_feature_dim()

            logger.info(
                f"ReID model loaded successfully. "
                f"Feature dim: {self._feature_dim}"
            )

        except ImportError:
            logger.error(
                "torchreid not installed. "
                "Install with: pip install torchreid "
                "or: cd deep-person-reid && python setup.py develop"
            )
            raise

        # Preprocessing transform (standard ImageNet normalization for ReID)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((config.REID_INPUT_HEIGHT, config.REID_INPUT_WIDTH)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def _get_feature_dim(self):
        """Determine the feature dimensionality by running a dummy input."""
        dummy = torch.randn(1, 3, config.REID_INPUT_HEIGHT, config.REID_INPUT_WIDTH)
        dummy = dummy.to(self.device)
        with torch.no_grad():
            feat = self.model(dummy)
        return feat.shape[1]

    @property
    def feature_dim(self):
        """Get the dimensionality of output feature vectors."""
        return self._feature_dim

    def extract(self, frame, bboxes):
        """
        Extract ReID embeddings for person crops.

        Args:
            frame: Full BGR frame (H, W, 3).
            bboxes: List of bounding boxes [[x1,y1,x2,y2], ...].

        Returns:
            np.ndarray of shape (N, feature_dim), L2-normalized embeddings.
            Returns empty array if no valid crops.
        """
        if len(bboxes) == 0:
            return np.zeros((0, self._feature_dim), dtype=np.float32)

        crops = self._prepare_crops(frame, bboxes)

        if len(crops) == 0:
            return np.zeros((0, self._feature_dim), dtype=np.float32)

        # Stack into batch tensor
        batch = torch.stack(crops).to(self.device)

        # Extract features
        with torch.no_grad():
            features = self.model(batch)

        # Convert to numpy and L2-normalize
        features = features.cpu().numpy()
        features = self._l2_normalize(features)

        return features

    def extract_single(self, frame, bbox):
        """
        Extract ReID embedding for a single person crop.

        Args:
            frame: Full BGR frame (H, W, 3).
            bbox: Bounding box [x1, y1, x2, y2].

        Returns:
            np.ndarray of shape (feature_dim,), L2-normalized embedding.
        """
        embeddings = self.extract(frame, [bbox])
        if len(embeddings) == 0:
            return np.zeros(self._feature_dim, dtype=np.float32)
        return embeddings[0]

    def _prepare_crops(self, frame, bboxes):
        """
        Crop and preprocess persons from the frame.

        Args:
            frame: BGR frame.
            bboxes: List of [x1, y1, x2, y2] bounding boxes.

        Returns:
            List of preprocessed torch tensors ready for the model.
        """
        h, w = frame.shape[:2]
        crops = []

        for bbox in bboxes:
            # Clip to frame boundaries
            x1 = max(0, int(bbox[0]))
            y1 = max(0, int(bbox[1]))
            x2 = min(w, int(bbox[2]))
            y2 = min(h, int(bbox[3]))

            # Skip invalid crops
            if x2 - x1 < 10 or y2 - y1 < 20:
                logger.debug(f"Skipping too-small crop: ({x2-x1}x{y2-y1})")
                continue

            # Crop the person
            crop = frame[y1:y2, x1:x2]

            # Convert BGR to RGB for the transform
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # Apply preprocessing transform
            try:
                tensor = self.transform(crop_rgb)
                crops.append(tensor)
            except Exception as e:
                logger.warning(f"Failed to preprocess crop: {e}")
                continue

        return crops

    @staticmethod
    def _l2_normalize(features):
        """
        L2-normalize feature vectors (each row normalized independently).

        Args:
            features: np.ndarray of shape (N, D).

        Returns:
            L2-normalized features of shape (N, D).
        """
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.maximum(norms, 1e-12)
        return features / norms

    @staticmethod
    def compute_similarity(feat1, feat2):
        """
        Compute cosine similarity between two L2-normalized feature vectors.

        Args:
            feat1: np.ndarray of shape (D,).
            feat2: np.ndarray of shape (D,).

        Returns:
            Cosine similarity (float).
        """
        return float(np.dot(feat1, feat2))
