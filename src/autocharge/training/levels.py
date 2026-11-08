
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import random


@dataclass
class Obstacle:
    kind: str
    x: float
    y: float
    size: float
    angle: float = 0.0


LEVELS = {
    1: {"count": 0, "types": []},
    2: {"count": 1, "types": ["box"]},
    3: {"count": 4, "types": ["cone"]},
    4: {"count": 4, "types": ["barrier"]},
    5: {"count": 5, "types": ["box", "cone", "barrier"]},
    6: {"count": 12, "types": ["stone", "box"]},
    7: {"count": 5, "types": ["slope", "box"]},
    8: {"count": 7, "types": ["slope", "stone"]},
    9: {"count": 12, "types": ["stone", "slope", "barrier"]},
    11: {"count": 3, "types": ["box", "stone", "slope"]},
    10: {"count": 16, "types": [
        "box", "cone", "barrier", "stone", "slope"
    ]},
}

SIZES = {
    "box": 0.22,
    "cone": 0.12,
    "barrier": 0.35,
    "stone": 0.06,
    "slope": 0.50,
}


def generate_level(level, seed=None, manual=None, mode="hybrid"):
    if level not in LEVELS:
        raise ValueError("Level must be between 1 and 11")

    if mode not in ("random", "manual", "hybrid"):
        raise ValueError("Invalid placement mode")

    rng = random.Random(seed)
    config = LEVELS[level]

    obstacles = (
        list(manual or [])
        if level >= 5 and mode != "random"
        else []
    )

    if mode != "manual":
        for _ in range(config["count"]):
            kind = rng.choice(config["types"])

            # Simple protected circular areas.
            for attempt in range(200):
                x = rng.uniform(-2.7, -0.5)
                y = rng.uniform(-1.2, 1.2)
                size = SIZES[kind]

                # Avoid the initial robot location.
                start_clear = (
                    (x + 3.0) ** 2 + (y - 0.5) ** 2
                ) ** 0.5 > size + 0.5

                # Avoid overlap with existing obstacles.
                separate = all(
                    ((x - o.x) ** 2 + (y - o.y) ** 2) ** 0.5
                    > size + o.size + 0.10
                    for o in obstacles
                )

                if start_clear and separate:
                    obstacles.append(
                        Obstacle(
                            kind=kind,
                            x=round(x, 3),
                            y=round(y, 3),
                            size=size,
                            angle=rng.choice([0, 15, 30, 45]),
                        )
                    )
                    break
            else:
                raise RuntimeError(
                    "Could not place obstacle safely"
                )

    return obstacles


def save_level(level, obstacles, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    content = {
        "level": level,
        "obstacles": [asdict(o) for o in obstacles],
    }

    path.write_text(json.dumps(content, indent=2))
