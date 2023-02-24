from phimaker import compute_ensemble

matrix = [
    (True, []),
    (True, []),
    (True, []),
    (False, []),
    (True, [0, 1]),
    (True, [0, 2]),
    (True, [1, 2]),
    (False, [0, 3]),
    (False, [1, 3]),
    (False, [2, 3]),
    (False, [4, 7, 8]),
    (False, [5, 7, 9]),
    (False, [6, 8, 9]),
    (True, [4, 5, 6]),
]

dgms = compute_ensemble(matrix)
print("f:")
print(dgms.f.unpaired)
print(dgms.f.paired)

print("g:")
print(dgms.g.unpaired)
print(dgms.g.paired)

print("im:")
print(dgms.im.unpaired)
print(dgms.im.paired)

print("ker:")
print(dgms.ker.unpaired)
print(dgms.ker.paired)

print("cok:")
print(dgms.cok.unpaired)
print(dgms.cok.paired)
