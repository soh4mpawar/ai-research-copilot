"""
Resumable Ingestion Checkpoint Manager (Phase 1 / PRD §6.1 / NFR).
Tracks batch ingestion state so pipeline resumes from last success on crash or restart.
"""

import json
import os
from typing import Set, Dict, Any


class IngestionCheckpoint:
    """State tracker for batch ingestion resumability."""

    def __init__(self, checkpoint_path: str = "data/metadata/ingestion_checkpoint.json"):
        self.checkpoint_path = checkpoint_path
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        self.completed_paper_ids: Set[str] = self._load_checkpoint()

    def _load_checkpoint(self) -> Set[str]:
        if os.path.exists(self.checkpoint_path):
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return set(data.get("completed_paper_ids", []))
            except Exception:
                pass
        return set()

    def is_completed(self, paper_id: str) -> bool:
        return paper_id in self.completed_paper_ids

    def mark_completed(self, paper_id: str):
        self.completed_paper_ids.add(paper_id)
        self._save_checkpoint()

    def _save_checkpoint(self):
        data = {
            "completed_paper_ids": list(self.completed_paper_ids),
            "total_completed": len(self.completed_paper_ids)
        }
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
