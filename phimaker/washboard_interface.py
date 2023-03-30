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
