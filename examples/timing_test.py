import sys
import os

sys.path.append(os.getcwd())

from gudhi import RipsComplex
import numpy as np
from phimaker import compute_ensemble
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import time

random.seed(42)
np.random.seed(42)

N = 200
N_nice = 20
max_diagram_dim = 1
jitter_strength = 0.05


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
times_parallel = []

# Compute diagrams
N = 5

print("Running in parallel")
for i in range(N):
    tic = time.time()
    dgms = compute_ensemble(matrix)
    toc = time.time()
    elapsed = toc - tic
    times_parallel.append(elapsed)
    print("Done")


times_serial = []
print("Running in serial")
for i in range(N):
    tic = time.time()
    dgms = compute_ensemble(matrix, num_threads=1)
    toc = time.time()
    elapsed = toc - tic
    times_serial.append(elapsed)
    print("Done")

print("Parallel")
print(np.mean(times_parallel))

print("Serial")
print(np.mean(times_serial))
