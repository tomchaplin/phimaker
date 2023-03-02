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

N = 10
aN_nice = 20
bN_nice = 20
max_diagram_dim = 2
jitter_strength = 0.05
a = 0.3
b = 0.8
truncation = 1.1


def get_jitter_torus_point(phase1, phase2, jit):
    random_phase1 = random.random() * jit
    random_phase2 = random.random() * jit
    phase1 = phase1 + random_phase1
    phase2 = phase2 + random_phase2
    angle1 = phase1 * 2 * math.pi
    angle2 = phase2 * 2 * math.pi
    return [
        (a * math.cos(angle1) + b) * math.cos(angle2),
        (a * math.cos(angle1) + b) * math.sin(angle2),
        a * math.sin(angle1),
    ]


def get_jitter_circle_point(phase, jit):
    random_phase = random.random() * jit
    return [
        b * math.cos(2 * math.pi * (phase + random_phase)),
        b * math.sin(2 * math.pi * (phase + random_phase)),
        0,
    ]


nice_points = np.array(
    [
        get_jitter_torus_point(i / aN_nice, j / bN_nice, jitter_strength)
        for i in range(aN_nice)
        for j in range(bN_nice)
    ]
)

filling_points = np.array(
    [get_jitter_circle_point(i / bN_nice, jitter_strength) for i in range(bN_nice)]
)


pts = np.vstack((nice_points, filling_points))

# Plot point
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(filling_points[:, 0], filling_points[:, 1], filling_points[:, 2])
ax.scatter(nice_points[:, 0], nice_points[:, 1], nice_points[:, 2])
ax.set_aspect("equal", adjustable="box")
plt.legend(["Filling points", "Torus points"], loc="lower right")
plt.savefig("fig/pointcloud_torus.png")
plt.show()


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
    annotated_col = (is_nice_smplx(smplx), sorted(sparse_bdry))
    matrix.append(annotated_col)
# Report
print("Got matrix")
print(len(matrix))
# Compute diagrams
dgms = compute_ensemble(matrix)
print("Done")


# Plot diagrams
def plot_diagram(
    diagram,
    entrance_times,
    dimensions,
    truncation,
    max_dim=2,
    ax=None,
    title=None,
    legend=True,
    dim_shift=0,
):
    all_pts = np.array(
        [
            [
                entrance_times[birth_idx],
                entrance_times[death_idx],
                dimensions[birth_idx] - dim_shift,
            ]
            for birth_idx, death_idx in diagram.paired
            if entrance_times[death_idx] != entrance_times[birth_idx]
            and dimensions[birth_idx] - dim_shift <= max_dim
        ]
        + [
            [
                entrance_times[birth_idx],
                truncation * 1.05,
                dimensions[birth_idx] - dim_shift,
            ]
            for birth_idx in diagram.unpaired
            if dimensions[birth_idx] - dim_shift <= max_dim
        ]
    )
    df = pd.DataFrame(data=all_pts, columns=["Birth", "Death", "Dimension"])
    ret_ax = sns.scatterplot(
        data=df, x="Birth", y="Death", hue="Dimension", ax=ax, legend=legend
    )
    ret_ax.set(xlabel=None)
    ret_ax.set(ylabel=None)
    ret_ax.set(title=title)
    sns.move_legend(ret_ax, "lower right")
    handle = ax if ax is not None else plt
    handle.plot([0, truncation * 1.05], [0, truncation * 1.05], "m", alpha=0.4)
    handle.plot(
        [0, 0, truncation * 1.05],
        [0, truncation * 1.05, truncation * 1.05],
        "m--",
        alpha=0.4,
    )


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=[3.5 * 3, 3.5 * 2])
plot_diagram(
    dgms.ker,
    entrance_times,
    dimensions,
    truncation,
    ax=axes[0][0],
    title="Kernel",
    dim_shift=1,
    max_dim=max_diagram_dim,
)
plot_diagram(
    dgms.rel,
    entrance_times,
    dimensions,
    truncation,
    ax=axes[0][1],
    title="Relative",
    max_dim=max_diagram_dim + 1,
)
plot_diagram(
    dgms.cok,
    entrance_times,
    dimensions,
    truncation,
    ax=axes[0][2],
    title="Cokernel",
    max_dim=max_diagram_dim,
)
plot_diagram(
    dgms.g,
    entrance_times,
    dimensions,
    truncation,
    ax=axes[1][0],
    title="Domain",
    max_dim=max_diagram_dim,
)
plot_diagram(
    dgms.im,
    entrance_times,
    dimensions,
    truncation,
    ax=axes[1][1],
    title="Image",
    max_dim=max_diagram_dim,
)
plot_diagram(
    dgms.f,
    entrance_times,
    dimensions,
    truncation,
    ax=axes[1][2],
    title="Codomain",
    legend=True,
    max_dim=max_diagram_dim,
)
plt.savefig("fig/5pack_torus.png")
