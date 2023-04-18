from pprint import pprint
from phimaker import compute_ensemble_cylinder

domain = [
    (0, 0, []),
    (0, 0, []),
    (0, 0, []),
    (0, 0, []),
    (1, 1, [0, 1]),
    (1, 1, [1, 2]),
    (1, 1, [2, 3]),
    (1, 1, [0, 3]),
    (10, 2, [4, 5, 6, 7]),
]

codomain = [
    (0, 0, []),
    (0, 0, []),
    (0, 0, []),
    (0, 0, []),
    (0.1, 1, [0, 1]),
    (0.1, 1, [1, 2]),
    (0.1, 1, [2, 3]),
    (0.1, 1, [0, 3]),
    (2, 2, [0, 2]),
    (2, 2, [4, 5, 8]),
    (2, 2, [6, 7, 8]),
]

map = [[0], [1], [2], [3], [4], [5], [6], [7], [9, 10]]

ensemble, metadata = compute_ensemble_cylinder(domain, codomain, map)
ker = ensemble.ker.paired

diagrams = {
    "f": ensemble.f,
    "g": ensemble.g,
    "rel": ensemble.rel,
    "ker": ensemble.ker,
    "im": ensemble.im,
    "cok": ensemble.cok,
}

for dig_name, dig in diagrams.items():
    print(dig_name)
    for pair in dig.paired:
        t0 = metadata.times[pair[0]]
        t1 = metadata.times[pair[1]]
        if t0 == t1:
            continue
        print(f"({t0}, {t1})")
    for idx in dig.unpaired:
        print(f"({metadata.times[idx]}, inf)")
