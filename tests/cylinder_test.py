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
ker = ensemble.ker.paired

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
    for pair in dgm.paired:
        t0 = metadata.times[pair[0]]
        t1 = metadata.times[pair[1]]
        if t0 == t1:
            continue
        print(f"({t0}, {t1})")
    for idx in dgm.unpaired:
        print(f"({metadata.times[idx]}, inf)")
