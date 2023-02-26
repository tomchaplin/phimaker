from gudhi import RipsComplex
import numpy as np
from phimaker import compute_ensemble
import math

N = 100

nice_points = np.array(
    [
        [0.5, 0],
        [0.5, 0.5],
        [0, 0.5],
        [-0.5, 0.5],
        [-0.5, 0],
        [-0.5, -0.5],
        [0, -0.5],
        [0.5, -0.5],
    ]
)

pts = np.vstack((nice_points, np.random.rand(N, 2) * 2 - 1))


def is_nice_point(idx):
    return idx < len(nice_points)


def is_nice_smplx(smplx):
    return all([is_nice_point(pt) for pt in smplx])


rcomp = RipsComplex(
    points=pts,
    max_edge_length=math.sqrt(2),
)
simplex_tree = rcomp.create_simplex_tree(max_dimension=2)
s_tree2 = simplex_tree.copy()
fmt = "%s -> %.2f"
for idx, f_val in enumerate(simplex_tree.get_filtration()):
    s_tree2.assign_filtration(f_val[0], idx)

matrix = []
for idx, f_val in enumerate(s_tree2.get_filtration()):
    smplx = f_val[0]
    sparse_bdry = [int(face_idx) for _, face_idx in s_tree2.get_boundaries(smplx)]
    annotated_col = (is_nice_smplx(smplx), sorted(sparse_bdry))
    matrix.append(annotated_col)

print("Got matrix")
print(len(matrix))

dgms = compute_ensemble(matrix)
print("Done")
#print("f:")
#print(dgms.f.unpaired)
#print(dgms.f.paired)
#
#print("g:")
#print(dgms.g.unpaired)
#print(dgms.g.paired)
#
#print("im:")
#print(dgms.im.unpaired)
#print(dgms.im.paired)
#
#print("ker:")
#print(dgms.ker.unpaired)
#print(dgms.ker.paired)
#
#print("cok:")
#print(dgms.cok.unpaired)
#print(dgms.cok.paired)
