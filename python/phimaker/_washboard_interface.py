from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.view import view_config
from pyramid.response import FileResponse
from pyramid.response import Response
from pathlib import Path
import threading
import signal
from contextlib import redirect_stdout
import io
import webbrowser


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


class WashboardServer:
    def __init__(self, washboard_obj=None, host="0.0.0.0", port=6543):
        self.washboard_obj = washboard_obj
        self.host = host
        self.port = port
        self.server = self._setup_server()

    def _setup_server(self):
        def data_route(request):
            return self.washboard_obj

        def main_page(request):
            index_file_path = (
                Path(__file__).parent / "washboard/dist/index.html"
            ).resolve()
            response = FileResponse(
                index_file_path,
                request=request,
                content_type="text/html",
            )
            return response

        with Configurator() as config:
            config.add_route("home", "/")
            config.add_view(main_page, route_name="home")
            config.add_route("data", "/data.json")
            config.add_view(data_route, renderer="json", route_name="data")
            config.scan()
            app = config.make_wsgi_app()
        return make_server(self.host, self.port, app)

    def serve_forever(self):
        self.server.serve_forever()

    def shutdown(self):
        self.server.shutdown()

    @staticmethod
    def _get_link(text):
        return f"\u001b]8;;{text}\u001b\\{text}\u001b]8;;\u001b\\"

    def open(self, open_browser=True):
        server_thread = threading.Thread(
            target=lambda: self.serve_forever(), name="WashboardServer"
        )
        server_thread.start()
        link = f"http://{self.host}:{self.port}"
        link_to_print = self._get_link(link)
        print(f"Hosting on {link_to_print}")

        def signal_handler(sig, frame):
            print("Closing down server")
            self.shutdown()

        signal.signal(signal.SIGINT, signal_handler)
        print("Press Ctrl+C to stop")

        if open_browser:
            webbrowser.open(link)

        signal.pause()

    @classmethod
    def build(
        cls,
        dgms,
        max_diagram_dim,
        truncation,
        dimensions,
        entrance_times,
        host="0.0.0.0",
        port=6543,
    ):
        obj = build_washboard_object(
            dgms, max_diagram_dim, truncation, dimensions, entrance_times
        )
        return cls(obj, host=host, port=port)
