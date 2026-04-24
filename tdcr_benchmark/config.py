from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class RobotConfig:
    segment_lengths: np.ndarray
    number_disks: np.ndarray
    subsegment_lengths: np.ndarray
    n_disk: int
    r_disk: float
    p_tendon: np.ndarray
    tensions: np.ndarray
    cc_tendon_displacements: np.ndarray
    youngs_modulus: float
    poissons_ratio: float
    shear_modulus: float
    outer_radius: float
    inner_radius: float
    moment_of_inertia: float
    gravity: float
    backbone_weight: np.ndarray
    disk_weight: np.ndarray
    external_force: np.ndarray
    external_moment: np.ndarray
    output_format: str


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "config.yaml"


def load_config(path: str | Path | None = None) -> RobotConfig:
    config_path = Path(path) if path is not None else default_config_path()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return parse_config(raw)


def parse_config(raw: dict[str, Any]) -> RobotConfig:
    robot = raw["robot"]
    segments = robot["segments"]
    if len(segments) != 2:
        raise ValueError("This benchmark currently supports exactly two segments.")

    number_disks = np.array([int(s["number_of_disks"]) for s in segments], dtype=int)
    if np.any(number_disks <= 0):
        raise ValueError("Each segment must have at least one disk.")

    segment_lengths = np.array([float(s["length"]) for s in segments], dtype=float)
    subsegment_lengths = np.concatenate(
        [
            np.full(number_disks[i], segment_lengths[i] / number_disks[i], dtype=float)
            for i in range(2)
        ]
    )
    n_disk = int(number_disks.sum())

    routing = robot["routing"]
    p_tendon = np.array(routing["tendon_positions"], dtype=float).T
    if p_tendon.shape != (4, 6):
        raise ValueError("This benchmark currently expects six homogeneous tendon positions.")

    tensions = _vector(robot.get("tensions", [0.0] * 6), 6, "robot.tensions")
    cc_tendon_displacements = _vector(
        robot.get("constant_curvature", {}).get("tendon_displacements", [0.0] * 6),
        6,
        "robot.constant_curvature.tendon_displacements",
    )

    backbone = robot["backbone"]
    youngs_modulus = float(backbone["youngs_modulus"])
    poissons_ratio = float(backbone["poissons_ratio"])
    shear_modulus = youngs_modulus / (2.0 * (1.0 + poissons_ratio))
    outer_radius = float(backbone["outer_radius"])
    inner_radius = float(backbone["inner_radius"])
    moment_of_inertia = 0.25 * np.pi * (outer_radius**4 - inner_radius**4)

    gravity = float(raw.get("environment", {}).get("gravity", 0.0))
    mass_per_length = float(backbone.get("mass_per_length", 0.0))
    disk_mass = float(robot.get("disk", {}).get("mass", 0.0))
    backbone_weight = mass_per_length * subsegment_lengths * gravity
    disk_weight = np.full(n_disk, disk_mass * gravity, dtype=float)

    external_loads = raw.get("external_loads", {})
    external_force = _vector(external_loads.get("force", [0.0, 0.0, 0.0, 0.0]), 4, "external_loads.force")
    external_moment = _vector(external_loads.get("moment", [0.0, 0.0, 0.0, 0.0]), 4, "external_loads.moment")

    return RobotConfig(
        segment_lengths=segment_lengths,
        number_disks=number_disks,
        subsegment_lengths=subsegment_lengths,
        n_disk=n_disk,
        r_disk=float(routing["distance_to_center"]),
        p_tendon=p_tendon,
        tensions=tensions,
        cc_tendon_displacements=cc_tendon_displacements,
        youngs_modulus=youngs_modulus,
        poissons_ratio=poissons_ratio,
        shear_modulus=shear_modulus,
        outer_radius=outer_radius,
        inner_radius=inner_radius,
        moment_of_inertia=moment_of_inertia,
        gravity=gravity,
        backbone_weight=backbone_weight,
        disk_weight=disk_weight,
        external_force=external_force,
        external_moment=external_moment,
        output_format=str(raw.get("output", {}).get("format", "csv")),
    )


def _vector(value: Any, size: int, name: str) -> np.ndarray:
    array = np.array(value, dtype=float).reshape(-1)
    if array.size != size:
        raise ValueError(f"{name} must contain exactly {size} values.")
    return array

