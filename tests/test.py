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
print("cod:")
print(dgms.cod.unpaired)
print(dgms.cod.paired)

print("dom:")
print(dgms.dom.unpaired)
print(dgms.dom.paired)

print("im:")
print(dgms.im.unpaired)
print(dgms.im.paired)

print("ker:")
print(dgms.ker.unpaired)
print(dgms.ker.paired)

print("cok:")
print(dgms.cok.unpaired)
print(dgms.cok.paired)

print("rel:")
print(dgms.rel.unpaired)
print(dgms.rel.paired)
