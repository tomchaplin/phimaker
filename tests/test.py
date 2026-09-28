import os
import sys

sys.path.append(os.getcwd())

from phimaker import sixpack_from_inclusion

matrix = [
    (True, 0, []),
    (True, 0, []),
    (True, 0, []),
    (False, 0, []),
    (True, 1, [0, 1]),
    (True, 1, [0, 2]),
    (True, 1, [1, 2]),
    (False, 1, [0, 3]),
    (False, 1, [1, 3]),
    (False, 1, [2, 3]),
    (False, 2, [4, 7, 8]),
    (False, 2, [5, 7, 9]),
    (False, 2, [6, 8, 9]),
    (True, 2, [4, 5, 6]),
]

dgms = sixpack_from_inclusion(matrix)
for name in ("cod", "dom", "im", "ker", "cok", "rel"):
    print(f"{name}:")
    print(getattr(dgms, name))
