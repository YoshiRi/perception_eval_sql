"""Configuration for GT-EST re-matching."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml

_SENTINEL = "__default__"


@dataclass
class RematchConfig:
    # ── Required ──────────────────────────────────────────────────────────
    csv_path: str
    output_dir: str

    # ── Run identity ──────────────────────────────────────────────────────
    label: str = "rematch"

    # ── Matching thresholds ───────────────────────────────────────────────
    # Per-class center-distance threshold (m). Use "__default__" as fallback.
    # e.g. {"car": 2.0, "pedestrian": 1.0, "__default__": 2.0}
    distance_thresholds_m: Dict[str, float] = field(
        default_factory=lambda: {_SENTINEL: 2.0}
    )

    # ── Label matching policy ─────────────────────────────────────────────
    # "strict"  : match only within the same label (GT car ↔ EST car)
    # "agnostic": match any GT to any EST regardless of label
    # "grouped" : match within user-defined groups (see label_groups)
    label_match: str = "strict"

    # Used only when label_match == "grouped".
    # Labels in the same inner list can be matched to each other.
    # e.g. [["car", "truck", "bus"], ["pedestrian"], ["bicycle", "motorbike"]]
    label_groups: Optional[List[List[str]]] = None

    # ── Pre-match filters ─────────────────────────────────────────────────
    # Minimum EST confidence to participate in matching (0.0 = no filter).
    confidence_min: float = 0.0

    # GT visibility values to exclude from the matching pool.
    # Excluded GTs are emitted as FN in the output regardless.
    visibility_exclude_from_gt: List[str] = field(default_factory=list)

    # ── Algorithm ─────────────────────────────────────────────────────────
    # "hungarian" : global optimal (recommended; matrices are tiny)
    # "greedy"    : sort EST by confidence descending, nearest-first
    algorithm: str = "hungarian"

    # ── Derived helpers ───────────────────────────────────────────────────

    def threshold_for(self, label: str) -> float:
        return self.distance_thresholds_m.get(
            label, self.distance_thresholds_m.get(_SENTINEL, 2.0)
        )

    def label_group_of(self, label: str) -> Optional[int]:
        """Return group index for a label, or None if not in any group."""
        if self.label_match != "grouped" or not self.label_groups:
            return None
        for i, grp in enumerate(self.label_groups):
            if label in grp:
                return i
        return None

    # ── Serialisation ─────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RematchConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, allow_unicode=True, sort_keys=False)

    # ── CLI ───────────────────────────────────────────────────────────────

    @staticmethod
    def add_args(parser) -> None:
        parser.add_argument("--config", help="Path to YAML config file")
        parser.add_argument("--csv", dest="csv_path", help="Override csv_path")
        parser.add_argument("--output-dir", help="Override output_dir")
        parser.add_argument("--label", help="Override label")
        parser.add_argument(
            "--threshold",
            type=float,
            metavar="M",
            help="Set uniform distance threshold in metres (overrides per-class config)",
        )
        parser.add_argument(
            "--label-match",
            choices=["strict", "agnostic", "grouped"],
            help="Override label_match policy",
        )
        parser.add_argument(
            "--confidence-min",
            type=float,
            help="Minimum EST confidence to participate in matching",
        )
        parser.add_argument(
            "--vis-exclude",
            help="Comma-separated visibility values to exclude from GT pool",
        )
        parser.add_argument(
            "--algorithm",
            choices=["hungarian", "greedy"],
            help="Override matching algorithm",
        )

    @classmethod
    def from_args(cls, args) -> "RematchConfig":
        if args.config:
            cfg = cls.from_yaml(args.config)
        else:
            if not args.csv_path or not args.output_dir:
                raise ValueError(
                    "Either --config or both --csv and --output-dir are required"
                )
            cfg = cls(csv_path=args.csv_path, output_dir=args.output_dir)

        if args.csv_path:
            cfg.csv_path = args.csv_path
        if args.output_dir:
            cfg.output_dir = args.output_dir
        if args.label:
            cfg.label = args.label
        if args.threshold is not None:
            cfg.distance_thresholds_m = {_SENTINEL: args.threshold}
        if args.label_match:
            cfg.label_match = args.label_match
        if args.confidence_min is not None:
            cfg.confidence_min = args.confidence_min
        if args.vis_exclude:
            cfg.visibility_exclude_from_gt = [
                v.strip() for v in args.vis_exclude.split(",")
            ]
        if args.algorithm:
            cfg.algorithm = args.algorithm

        return cfg
