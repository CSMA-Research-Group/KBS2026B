"""Constructed loaded-motion cases; policy inputs contain no reference labels."""
import random

from innovation_stage3.storage import stable
from e3_cpu_correctness.core import SCHEMA, validate_world


def member(family, index):
    if type(index) is not int or not 0 <= index < family["count"]:
        raise ValueError("Member outside declared family")
    x, y = family["origin"]
    return {"goal_id": family["id"] + ":" + str(index),
            "configuration_id": family["configuration_id"],
            "target": [x + index % family["columns"], y + index // family["columns"]]}


def member_at(family, xy):
    dx, dy = xy[0] - family["origin"][0], xy[1] - family["origin"][1]
    if dx < 0 or dy < 0 or dx >= family["columns"]:
        return None
    index = dy * family["columns"] + dx
    return index if index < family["count"] else None


def bounds(family):
    x, y = family["origin"]
    return x, y, x + min(family["columns"], family["count"]) - 1, y + (family["count"] - 1) // family["columns"]


def validate_public(public):
    if set(public) != {"world", "families", "contract"} or public["contract"] != "unique_loaded_goal":
        raise ValueError("Unsupported public decision contract")
    world = public["world"]
    validate_world(world)
    configs = {c["id"] for c in world["configurations"]}
    ids, goals = set(), set()
    for f in public["families"]:
        if set(f) != {"id", "configuration_id", "origin", "columns", "count"}:
            raise ValueError("Unexpected family fields")
        if not isinstance(f["id"], str) or f["id"] in ids or f["configuration_id"] not in configs:
            raise ValueError("Invalid family identity")
        ids.add(f["id"])
        if type(f["count"]) is not int or f["count"] < 1 or type(f["columns"]) is not int or not 1 <= f["columns"] <= f["count"]:
            raise ValueError("Invalid family extent")
        if not isinstance(f["origin"], list) or len(f["origin"]) != 2 or any(type(v) is not int for v in f["origin"]):
            raise ValueError("Invalid family origin")
        lo_x, lo_y, hi_x, hi_y = bounds(f)
        if lo_x < 0 or lo_y < 0 or hi_x >= world["width"] or hi_y >= world["height"]:
            raise ValueError("Family extends outside world")
        for i in range(f["count"]):
            m = member(f, i)
            key = m["configuration_id"], tuple(m["target"])
            if key in goals:
                raise ValueError("Repeated physical candidate")
            goals.add(key)
    if not goals:
        raise ValueError("This experiment requires a nonempty semantic candidate family")


def construct(count, feasible, obstruction, replicate, seed):
    if count < 4 or feasible not in (0, 1, 2) or obstruction not in ("shared", "unshared"):
        raise ValueError("Unsupported design cell")
    # Two spacious chambers with two separate, one-cell-wide bent channels.
    # A point relaxation traverses each bend; elongated objects cannot rotate
    # there under the declared square-sweep motion model. The shared condition
    # additionally closes both channels, enabling one family-level exclusion.
    width, height = 41, 31
    free = {(x, y) for x in list(range(1, 14)) + list(range(27, 40)) for y in range(1, 30)}
    upper, lower = (19, 22) if replicate % 2 == 0 else (21, 18)
    free.update((x, 7) for x in range(13, upper + 1))
    free.update((upper, y) for y in range(7, 11))
    free.update((x, 10) for x in range(upper, 28))
    free.update((x, 23) for x in range(13, lower + 1))
    free.update((lower, y) for y in range(20, 24))
    free.update((x, 20) for x in range(lower, 28))
    if obstruction == "shared":
        free.remove((16, 7))
        free.remove((16, 23))
    configs = [{"id": "c" + str(i), "footprint": fp,
                "start": [x, y, (i + replicate) % 2]}
               for i, (fp, (x, y)) in enumerate(zip(
                   ([3, 1], [1, 3], [5, 1], [1, 5]), ((5, 5), (7, 12), (6, 19), (8, 25))))]
    world = {"schema": SCHEMA, "width": width, "height": height,
             "blocked": [[x, y] for y in range(height) for x in range(width) if (x, y) not in free],
             "configurations": configs, "rotation": "square_sweep"}
    families = []
    negatives = count - feasible
    for i, c in enumerate(configs):
        number = negatives // 4 + (i < negatives % 4)
        if number:
            families.append({"configuration_id": c["id"], "origin": [30, 4],
                             "columns": min(7, number), "count": number})
    for i in range(feasible):
        c = configs[(i + replicate) % 4]
        families.append({"configuration_id": c["id"], "origin": [10, c["start"][1]], "columns": 1, "count": 1})
    random.Random(stable([seed, "families", count, feasible, obstruction, replicate])).shuffle(families)
    for i, f in enumerate(families):
        f["id"] = "f" + str(i)
    public = {"world": world, "families": families, "contract": "unique_loaded_goal"}
    validate_public(public)
    metadata = {"candidate_count": count, "designed_feasible_count": feasible,
                "obstruction": obstruction, "replicate": replicate,
                "layout_cluster": obstruction + "_" + str(replicate),
                "scale_cluster": str(feasible) + "_" + obstruction + "_" + str(replicate)}
    return {"case_id": "e3b_" + stable([seed, metadata])[:16], "metadata": metadata, "public": public}


def design(config):
    return [construct(n, k, structure, r, config["world_seed"])
            for n in config["candidate_counts"] for k in config["feasible_counts"]
            for structure in config["obstructions"] for r in range(config["construction_replicates"])]
