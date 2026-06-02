"""Configuration for safety-critical detection metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class SafetyConfig:
    # ── Required ──────────────────────────────────────────────────────────
    csv_path: str
    output_dir: str

    # ── Run identity ──────────────────────────────────────────────────────
    label: str = "eval"

    # ── Evaluation scope ──────────────────────────────────────────────────
    classes: List[str] = field(
        default_factory=lambda: ["car", "truck", "bus", "pedestrian"]
    )
    # Upper bound of distance range to evaluate (metres).
    # Must align with bin edges in the CSV: 20, 40, 60, 80, 100, 120, 140, 160, 180, 200.
    dist_max_m: int = 100

    # Visibility values to exclude from the GT denominator.
    # NONE = fully occluded objects that are physically undetectable.
    visibility_exclude: List[str] = field(default_factory=lambda: ["NONE"])

    # ── nuScenes matching ─────────────────────────────────────────────────
    # Center-distance thresholds (m) used to re-judge TP/FP from ATE.
    match_thresholds_m: List[float] = field(
        default_factory=lambda: [0.5, 1.0, 2.0, 4.0]
    )

    # ── Outlier handling ──────────────────────────────────────────────────
    # Hard cap on |speed_error| before computing mAVE (sensor artifact filter).
    speed_error_clip_mps: float = 50.0

    # ── NDS normalization ─────────────────────────────────────────────────
    # Scores are: 1 - min(metric / norm, 1.0)
    ate_norm_m: float = 2.0
    aoe_norm_rad: float = math.pi
    ave_norm_mps: float = 10.0
    # Weight of mAP term in NDS formula.
    map_weight_in_nds: int = 5

    # ── Optional per-class safety weights ─────────────────────────────────
    # If set, mAP = weighted mean of per-class APs.
    # Useful for prioritising pedestrian/cyclist over car.
    class_weights: Optional[Dict[str, float]] = None

    # ── Derived (not serialised) ──────────────────────────────────────────

    def r_bins(self) -> List[str]:
        """Return CSV r-bin strings that fall within dist_max_m."""
        all_bins = [
            "0-20", "20-40", "40-60", "60-80", "80-100",
            "100-120", "120-140", "140-160", "160-180", "180-200",
        ]
        result = []
        for b in all_bins:
            start = int(b.split("-")[0])
            if start < self.dist_max_m:
                result.append(b)
        return result

    def nds_n_terms(self) -> int:
        """Total denominator of NDS: map_weight + number of TP metrics."""
        return self.map_weight_in_nds + 3  # ATE, AOE, AVE

    # ── Serialisation ─────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SafetyConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        d = asdict(self)
        # round-trip floats cleanly
        with open(path, "w") as f:
            yaml.dump(d, f, allow_unicode=True, sort_keys=False)

    # ── CLI override ──────────────────────────────────────────────────────

    @staticmethod
    def add_args(parser) -> None:
        """Register override flags onto an argparse.ArgumentParser."""
        parser.add_argument("--config", help="Path to YAML config file")
        parser.add_argument("--csv", dest="csv_path", help="Override csv_path")
        parser.add_argument("--output-dir", help="Override output_dir")
        parser.add_argument("--label", help="Override label for this run")
        parser.add_argument(
            "--classes",
            help="Comma-separated class list, e.g. car,truck,pedestrian",
        )
        parser.add_argument(
            "--dist-max", dest="dist_max_m", type=int,
            help="Override dist_max_m (must be multiple of 20, max 200)",
        )
        parser.add_argument(
            "--vis-exclude",
            help="Comma-separated visibility values to exclude, e.g. NONE",
        )
        parser.add_argument(
            "--match-thresholds",
            help="Comma-separated matching thresholds in metres, e.g. 0.5,1.0,2.0,4.0",
        )
        parser.add_argument(
            "--speed-clip", dest="speed_error_clip_mps", type=float,
            help="Override speed_error_clip_mps",
        )

    @classmethod
    def from_args(cls, args) -> "SafetyConfig":
        """Build config from parsed CLI args (config YAML + overrides)."""
        if args.config:
            cfg = cls.from_yaml(args.config)
        else:
            # Require at minimum csv_path and output_dir
            if not args.csv_path or not args.output_dir:
                raise ValueError(
                    "Either --config or both --csv and --output-dir are required"
                )
            cfg = cls(csv_path=args.csv_path, output_dir=args.output_dir)

        # Apply CLI overrides
        if args.csv_path:
            cfg.csv_path = args.csv_path
        if args.output_dir:
            cfg.output_dir = args.output_dir
        if args.label:
            cfg.label = args.label
        if args.classes:
            cfg.classes = [c.strip() for c in args.classes.split(",")]
        if args.dist_max_m:
            cfg.dist_max_m = args.dist_max_m
        if args.vis_exclude:
            cfg.visibility_exclude = [v.strip() for v in args.vis_exclude.split(",")]
        if args.match_thresholds:
            cfg.match_thresholds_m = [
                float(t.strip()) for t in args.match_thresholds.split(",")
            ]
        if args.speed_error_clip_mps:
            cfg.speed_error_clip_mps = args.speed_error_clip_mps

        return cfg
