import sys
import os

sys.path.append(os.getcwd())

from gudhi import RipsComplex
import numpy as np
from phimaker import compute_ensemble
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
    annotated_col = (is_nice_smplx(smplx), sorted(sparse_bdry))
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


# Create washboard json file


def build_washboard_object(
    dgms, max_diagram_dim, truncation, dimensions, entrance_times
):
    def make_pairing_obj(dgm, additional_max_dim=0, dim_shift=0):
        pairings = {"paired": list(dgm.paired), "unpaired": list(dgm.unpaired)}
        pairings = remove_trivial_pairings(pairings)
        pairings = filter_by_dimension(pairings, additional_max_dim, dim_shift)
        pairings = group_by_dimension(pairings, additional_max_dim, dim_shift)
        return pairings

    # Remove pairings with 0 lifetime
    def remove_trivial_pairings(pairings):
        new_pairings = []
        for pairing in pairings["paired"]:
            if entrance_times[pairing[0]] == entrance_times[pairing[1]]:
                continue
            new_pairings.append(pairing)
        pairings["paired"] = new_pairings
        return pairings

    # Remove pairings in too high a dimension
    def filter_by_dimension(pairings, additional_max_dim, dim_shift):
        new_paired = []
        new_unpaired = []
        for pairing in pairings["paired"]:
            dim = dimensions[pairing[0]] - dim_shift
            if dim > max_diagram_dim + additional_max_dim:
                continue
            new_paired.append(pairing)
        for unpaired in pairings["unpaired"]:
            dim = dimensions[unpaired] - dim_shift
            # Only add additional_max_dim to pairings
            if dim > max_diagram_dim:
                continue
            new_unpaired.append(unpaired)
        pairings["paired"] = new_paired
        pairings["unpaired"] = new_unpaired
        return pairings

    def group_by_dimension(pairings, additional_max_dim, dim_shift):
        diagrams_by_dimension = [
            [] for _ in range(max_diagram_dim + additional_max_dim + 1)
        ]
        for pairing in pairings["paired"]:
            dim = dimensions[pairing[0]] - dim_shift
            diagrams_by_dimension[dim].append(pairing)
        for unpaired in pairings["unpaired"]:
            dim = dimensions[unpaired] - dim_shift
            diagrams_by_dimension[dim].append([unpaired])
        return diagrams_by_dimension

    def add_empty_relations(all_pairings, key):
        base_pairings = all_pairings[key]
        for d, diagram_d in enumerate(base_pairings):
            for idx, pairing in enumerate(diagram_d):
                base_pairings[d][idx] = [pairing, {}]
        return base_pairings

    def fill_relations(all_pairings, key):
        base_pairings = all_pairings[key]
        for d, diagram_d in enumerate(base_pairings):
            for idx, pairing_relations in enumerate(diagram_d):
                pairing = pairing_relations[0]
                relations = compute_relations(all_pairings, key, pairing)
                base_pairings[d][idx] = [pairing, relations]
        return base_pairings

    def compute_relations(all_pairings, base_key, pairing):
        relations = {}
        for key in all_pairings.keys():
            if key == base_key:
                continue
            these_relations = []
            diagram = all_pairings[key]
            for d, diagram_d in enumerate(diagram):
                for idx, other_pairing in enumerate(diagram_d):
                    if not set(pairing).isdisjoint(other_pairing[0]):
                        these_relations.append([d, idx])
            relations[key] = these_relations
        return relations

    def replace_with_f_times(all_pairings):
        for _, diagrams in all_pairings.items():
            for diagram in diagrams:
                for pairing_relations in diagram:
                    pairing = pairing_relations[0]
                    coords = [entrance_times[idx] for idx in pairing]
                    pairing_relations[0] = coords
        return all_pairings

    obj = {}
    obj["max_dim"] = max_diagram_dim
    obj["pseudo_inf"] = truncation * 1.05
    pairings = {
        "codomain": make_pairing_obj(dgms.f),
        "domain": make_pairing_obj(dgms.g),
        "image": make_pairing_obj(dgms.im),
        "kernel": make_pairing_obj(dgms.ker, dim_shift=1),
        "cokernel": make_pairing_obj(dgms.cok),
        "relative": make_pairing_obj(dgms.rel, additional_max_dim=1),
    }
    for key in pairings.keys():
        pairings[key] = add_empty_relations(pairings, key)
    for key in pairings.keys():
        pairings[key] = fill_relations(pairings, key)
    obj["pairings"] = replace_with_f_times(pairings)
    return obj


washboard_object = build_washboard_object(
    dgms, max_diagram_dim, truncation, dimensions, entrance_times
)
json_str = json.dumps(washboard_object)

with open("data.json", "w") as f:
    f.write(json_str)
