"""
In-memory person gallery for Re-Identification.

Stores person embeddings, manages global ID assignment, and handles
matching of new detections against known persons using cosine similarity.
Supports EMA-smoothed embedding updates for robust re-identification.
"""

import logging
import time

import numpy as np

import config
from utils import cosine_similarity

logger = logging.getLogger(__name__)


class PersonEntry:
    """
    A single person entry in the gallery.

    Attributes:
        global_id: Unique global person ID.
        embedding: EMA-smoothed L2-normalized feature vector.
        first_seen: Timestamp of first identification.
        last_seen: Timestamp of most recent identification.
        appearances: Number of times this person has been identified.
        best_confidence: Highest detection confidence seen.
        is_active: Whether this person is currently visible.
        re_entries: Number of times this person re-entered after leaving.
    """

    def __init__(self, global_id, embedding):
        self.global_id = global_id
        self.embedding = np.array(embedding, dtype=np.float32).copy()
        self.first_seen = time.time()
        self.last_seen = self.first_seen
        self.appearances = 1
        self.best_confidence = 0.0
        self.is_active = True
        self.re_entries = 0

    def __repr__(self):
        return (
            f"Person(id={self.global_id}, appearances={self.appearances}, "
            f"re_entries={self.re_entries}, active={self.is_active})"
        )


class Gallery:
    """
    In-memory gallery of known persons.

    Maintains a collection of PersonEntry objects and provides methods
    for matching new embeddings against the gallery, assigning global IDs,
    and updating embeddings with exponential moving average smoothing.

    All data is stored in RAM — no database required.

    Usage:
        gallery = Gallery()
        global_id, is_new = gallery.match_and_assign(embedding)
        gallery.update_embedding(global_id, new_embedding)
        stats = gallery.get_stats()
    """

    def __init__(self):
        self._entries = {}       # global_id -> PersonEntry
        self._next_id = 1        # Next available global ID
        self._total_re_entries = 0

        logger.info("Gallery initialized (RAM-only mode)")

    def match_and_assign(self, embedding, confidence=0.0):
        """
        Match an embedding against the gallery and assign a global ID.

        If the best match exceeds GALLERY_MATCH_THRESHOLD, the existing
        person's global ID is returned. Otherwise, a new person entry
        is created.

        Args:
            embedding: L2-normalized feature vector, shape (D,).
            confidence: Detection confidence score.

        Returns:
            Tuple of (global_id, is_new_person).
                global_id: The assigned global person ID.
                is_new_person: True if this is a newly created person.
        """
        if len(self._entries) == 0:
            return self._create_new_person(embedding, confidence), True

        # Compute similarity against all gallery entries
        best_id = None
        best_similarity = -1.0

        for gid, entry in self._entries.items():
            sim = cosine_similarity(embedding, entry.embedding)
            if sim > best_similarity:
                best_similarity = sim
                best_id = gid

        if best_similarity >= config.GALLERY_MATCH_THRESHOLD and best_id is not None:
            # Matched existing person
            entry = self._entries[best_id]

            # Check if this is a re-entry (was inactive)
            if not entry.is_active:
                entry.re_entries += 1
                self._total_re_entries += 1
                logger.info(
                    f"Re-entry detected! Person {best_id} returned "
                    f"(similarity: {best_similarity:.3f}, "
                    f"re-entry #{entry.re_entries})"
                )

            entry.is_active = True
            entry.last_seen = time.time()
            entry.appearances += 1
            if confidence > entry.best_confidence:
                entry.best_confidence = confidence

            logger.debug(
                f"Matched person {best_id} "
                f"(similarity: {best_similarity:.3f})"
            )
            return best_id, False

        else:
            # No match — new person
            logger.debug(
                f"No match found (best similarity: {best_similarity:.3f}). "
                f"Creating new person."
            )
            return self._create_new_person(embedding, confidence), True

    def _create_new_person(self, embedding, confidence=0.0):
        """
        Create a new person entry in the gallery.

        Args:
            embedding: L2-normalized feature vector.
            confidence: Detection confidence.

        Returns:
            The new global ID.
        """
        gid = self._next_id
        self._next_id += 1

        entry = PersonEntry(gid, embedding)
        entry.best_confidence = confidence
        self._entries[gid] = entry

        logger.info(f"New person created: Global ID {gid}")

        # Prune gallery if max size is set
        if config.GALLERY_MAX_SIZE > 0 and len(self._entries) > config.GALLERY_MAX_SIZE:
            self._prune_oldest()

        return gid

    def update_embedding(self, global_id, new_embedding):
        """
        Update a person's embedding using Exponential Moving Average (EMA).

        New embedding = α * old_embedding + (1 - α) * new_embedding
        Then re-normalize to unit length.

        Args:
            global_id: The person's global ID.
            new_embedding: New L2-normalized feature vector, shape (D,).
        """
        if global_id not in self._entries:
            logger.warning(f"Cannot update unknown person {global_id}")
            return

        entry = self._entries[global_id]
        alpha = config.GALLERY_EMA_ALPHA

        # EMA update
        entry.embedding = alpha * entry.embedding + (1 - alpha) * new_embedding

        # Re-normalize to unit length
        norm = np.linalg.norm(entry.embedding)
        if norm > 1e-12:
            entry.embedding /= norm

        entry.last_seen = time.time()

    def mark_inactive(self, global_id):
        """
        Mark a person as no longer visible (left the frame).

        Args:
            global_id: The person's global ID.
        """
        if global_id in self._entries:
            self._entries[global_id].is_active = False

    def mark_all_inactive(self):
        """Mark all persons as inactive (start of frame, before matching)."""
        for entry in self._entries.values():
            entry.is_active = False

    def get_unique_count(self):
        """Get the total number of unique persons ever seen."""
        return len(self._entries)

    def get_active_count(self):
        """Get the number of currently visible persons."""
        return sum(1 for e in self._entries.values() if e.is_active)

    def get_total_re_entries(self):
        """Get the total number of re-entry events."""
        return self._total_re_entries

    def get_person(self, global_id):
        """Get a person entry by global ID."""
        return self._entries.get(global_id)

    def get_all_persons(self):
        """Get all person entries."""
        return dict(self._entries)

    def get_stats(self):
        """
        Get gallery statistics.

        Returns:
            Dict with:
                unique_count: Total unique persons
                active_count: Currently visible persons
                re_entries: Total re-entry events
                gallery_size: Current gallery size
        """
        return {
            "unique_count": self.get_unique_count(),
            "active_count": self.get_active_count(),
            "re_entries": self.get_total_re_entries(),
            "gallery_size": len(self._entries),
        }

    def _prune_oldest(self):
        """Remove the oldest (by last_seen) entry to stay within max gallery size."""
        if len(self._entries) <= config.GALLERY_MAX_SIZE:
            return

        oldest_id = min(
            self._entries,
            key=lambda gid: self._entries[gid].last_seen,
        )
        del self._entries[oldest_id]
        logger.debug(f"Pruned oldest gallery entry: {oldest_id}")

    def reset(self):
        """Clear the entire gallery."""
        self._entries.clear()
        self._next_id = 1
        self._total_re_entries = 0
        logger.info("Gallery reset")
