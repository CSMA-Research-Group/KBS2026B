"""Third, independently written physical model (research audit).

Written from the documented rules only (innovation_stage2/domain.py docstring and the
paper's description), without importing any project module:

carrying  robot is a point while empty-handed; it moves between 4-neighbour free cells;
          it may pick a movable object whose mass <= payload when standing on the object's
          anchor cell and the object's footprint (at its own orientation) is free; while
          holding, a move requires the union of the current and next footprints to be free;
          a quarter-turn requires the centred square of side max(footprint) to be free;
          placing on target t requires standing on t's cell with an allowed orientation.
          Other objects never obstruct (as in the project model).  Colour compatibility is
          NOT physical.  Place is terminal, so goals are (object, target) classes.
workflow  STRIPS-like closure from the initial fact set; a goal is feasible iff some
          reachable state contains all its required facts.

Structure deliberately differs from the project's searches: carrying computes one
empty-handed reachability set, then one carried-configuration closure per pickable object,
and reads off every target at once (a shared, class-level certificate per object).
Returns feasible goal sets plus a work count in a single unit (configurations expanded).
"""
from collections import deque


def _cells(fp, x, y, o):
    ex = fp[o] // 2
    ey = fp[1 - o] // 2
    return [(x + dx, y + dy) for dy in range(-ey, ey + 1) for dx in range(-ex, ex + 1)]


def carrying_feasible(runtime, sources=None):
    """Feasible 'objectID->targetID' goals; `sources` restricts which objects are explored."""
    g = runtime['grid']
    W, H = g['width'], g['height']
    blocked = {tuple(c) for c in g['blocked']}
    free = lambda c: 0 <= c[0] < W and 0 <= c[1] < H and c not in blocked
    start = tuple(runtime['robot']['start'])
    seen = {start}
    q = deque([start])
    work = 0
    while q:
        x, y = q.popleft()
        work += 1
        for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if n not in seen and free(n) and free((x, y)):
                seen.add(n)
                q.append(n)
    feasible, per_object_work = set(), {}
    for obj in runtime['objects']:
        if sources is not None and obj['id'] not in sources:
            continue
        w0 = work
        cell = tuple(obj['cell'])
        fp = obj['footprint']
        ok = (obj['movable'] and obj['mass'] <= runtime['robot']['payload'] and cell in seen
              and all(free(c) for c in _cells(fp, cell[0], cell[1], obj['orientation'])))
        reach = set()
        if ok:
            s0 = (cell[0], cell[1], obj['orientation'])
            reach = {s0}
            q = deque([s0])
            side = max(fp)
            while q:
                x, y, o = q.popleft()
                work += 1
                here = set(_cells(fp, x, y, o))
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    n = (nx, ny, o)
                    if n not in reach and all(free(c) for c in here | set(_cells(fp, nx, ny, o))):
                        reach.add(n)
                        q.append(n)
                r = side // 2
                n = (x, y, 1 - o)
                if n not in reach and all(free((x + dx, y + dy)) for dx in range(-r, r + 1) for dy in range(-r, r + 1)):
                    reach.add(n)
                    q.append(n)
        for t in runtime['targets']:
            tx, ty = t['cell']
            if any((tx, ty, o) in reach for o in t['allowed_orientations']):
                feasible.add(obj['id'] + '->' + t['id'])
        per_object_work[obj['id']] = work - w0
    return feasible, {'configurations_expanded': work, 'per_object': per_object_work}


def workflow_feasible(runtime):
    init = frozenset(runtime['initial'])
    seen = {init}
    q = deque([init])
    work = 0
    while q:
        s = q.popleft()
        for a in runtime['actions']:
            work += 1
            if set(a['requires']) <= s:
                n = frozenset((s - set(a['delete'])) | set(a['add']))
                if n not in seen:
                    seen.add(n)
                    q.append(n)
    feasible = {gl['id'] for gl in runtime['goals'] if any(set(gl['requires']) <= s for s in seen)}
    return feasible, {'action_tests': work, 'closed_states': len(seen)}


def feasible(public):
    if public['domain'] == 'carrying':
        return carrying_feasible(public['runtime'])
    return workflow_feasible(public['runtime'])
