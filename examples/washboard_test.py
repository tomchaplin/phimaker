import sys
import os

sys.path.append(os.getcwd())

from gudhi import RipsComplex
import numpy as np
from phimaker import compute_ensemble
from phimaker import WashboardServer
import math
import matplotlib.pyplot as plt
import random
import json

N = 100
N_nice = 20
max_diagram_dim = 1
jitter_strength = 0.05

random.seed(42)
np.random.seed(42)


def get_jitterer_circle_point(phase, jit):
    random_phase = random.random() * jit
    return [
        0.7 * math.cos(2 * math.pi * (phase + random_phase)),
        0.7 * math.sin(2 * math.pi * (phase + random_phase)),
    ]


nice_points = np.array(
    [get_jitterer_circle_point(i / N_nice, jitter_strength) for i in range(N_nice)]
)

random_points = np.random.rand(N, 2) * 2 - 1

pts = np.vstack((nice_points, random_points))

truncation = math.sqrt(2)


def is_nice_point(idx):
    return idx < len(nice_points)


def is_nice_smplx(smplx):
    return all([is_nice_point(pt) for pt in smplx])


rcomp = RipsComplex(points=pts, max_edge_length=truncation)


# Build simplex tree
simplex_tree = rcomp.create_simplex_tree(max_dimension=max_diagram_dim + 1)
# Build second simplex tree with index as filtration value
s_tree2 = simplex_tree.copy()
entrance_times = []
dimensions = []
for idx, f_val in enumerate(simplex_tree.get_filtration()):
    s_tree2.assign_filtration(f_val[0], idx)
    entrance_times.append(f_val[1])
    dimensions.append(len(f_val[0]) - 1)
# Build up matrix to pass to phimaker
matrix = []
for idx, f_val in enumerate(s_tree2.get_filtration()):
    smplx = f_val[0]
    sparse_bdry = [int(face_idx) for _, face_idx in s_tree2.get_boundaries(smplx)]
    if len(sparse_bdry) == 0:
        dimension = 0
    else:
        dimension = len(sparse_bdry) - 1
    annotated_col = (is_nice_smplx(smplx), dimension, sorted(sparse_bdry))
    matrix.append(annotated_col)
# Report
print("Got matrix")
print(len(matrix))
# Compute diagrams
dgms = compute_ensemble(matrix)
print("Done")
# Plot point
plt.scatter(random_points[:, 0], random_points[:, 1])
plt.scatter(nice_points[:, 0], nice_points[:, 1])
plt.legend(["Random points", "Even points"], loc="lower right")
plt.savefig("fig/pointcloud.png")


# Serve washboard
washboard_server = WashboardServer.build(
    dgms,
    max_diagram_dim,
    truncation,
    dimensions,
    entrance_times,
    host="0.0.0.0",
    port=6543,
)
washboard_server.open()
