"""Configuration for Distance-To-Vehicle weighted mAP calculation."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class WeightedMapConfig:
    # ── Required ──────────────────────────────────────────────────────────
    csv_path: str
    output_dir: str

    # ── Run identity ──────────────────────────────────────────────────────
    label: str = "weighted_map"

    # ── Evaluation scope ──────────────────────────────────────────────────
    classes: List[str] = field(
        default_factory=lambda: ["car", "truck", "bus", "pedestrian"]
    )
    max_distance_m: float = 100.0

    # Visibility values to exclude from GT (weight set to 0).
    visibility_exclude: List[str] = field(default_factory=lambda: ["NONE"])

    # ── Distance weight ───────────────────────────────────────────────────
    # weight = exp(-lambda * DTV)
    distance_weight_lambda: float = 0.01

    # ── FOV filter ────────────────────────────────────────────────────────
    # Objects outside this FOV (centred on ego forward = +x axis) get weight=0.
    forward_fov_deg: float = 240.0

    # ── FP weighting ─────────────────────────────────────────────────────
    # "est_distance": fp_weight = in_fov_est * exp(-lambda * DTV_est)
    # "uniform":      fp_weight = 1.0
    fp_weighting: str = "est_distance"

    # ── AP method ─────────────────────────────────────────────────────────
    # "stepwise": Σ Δrecall * precision  (sklearn-style)
    ap_method: str = "stepwise"

    # ── Serialisation ─────────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WeightedMapConfig":
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
            "--classes",
            help="Comma-separated class list, e.g. car,truck,pedestrian",
        )
        parser.add_argument(
            "--dist-max", dest="max_distance_m", type=float,
            help="Override max_distance_m",
        )
        parser.add_argument(
            "--lambda", dest="distance_weight_lambda", type=float,
            help="Override distance_weight_lambda",
        )
        parser.add_argument(
            "--fov", dest="forward_fov_deg", type=float,
            help="Override forward_fov_deg",
        )
        parser.add_argument(
            "--fp-weighting", choices=["est_distance", "uniform"],
            help="Override fp_weighting",
        )
        parser.add_argument(
            "--vis-exclude",
            help="Comma-separated visibility values to exclude, e.g. NONE",
        )

    @classmethod
    def from_args(cls, args) -> "WeightedMapConfig":
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
        if args.classes:
            cfg.classes = [c.strip() for c in args.classes.split(",")]
        if args.max_distance_m:
            cfg.max_distance_m = args.max_distance_m
        if args.distance_weight_lambda:
            cfg.distance_weight_lambda = args.distance_weight_lambda
        if args.forward_fov_deg:
            cfg.forward_fov_deg = args.forward_fov_deg
        if args.fp_weighting:
            cfg.fp_weighting = args.fp_weighting
        if args.vis_exclude:
            cfg.visibility_exclude = [v.strip() for v in args.vis_exclude.split(",")]

        return cfg
