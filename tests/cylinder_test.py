from phimaker import sixpack

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

ensemble, metadata = sixpack(domain, codomain, map)
ker = ensemble.ker

diagrams = {
    "cod": ensemble.cod,
    "dom": ensemble.dom,
    "rel": ensemble.rel,
    "ker": ensemble.ker,
    "im": ensemble.im,
    "cok": ensemble.cok,
}

for dgm_name, dgm in diagrams.items():
    print(dgm_name)
    for birth, death in dgm.items():
        t0 = metadata.times[birth]
        if death is None:
            print(f"({t0}, inf)")
            continue
        t1 = metadata.times[death]
        if t0 != t1:
            print(f"({t0}, {t1})")
