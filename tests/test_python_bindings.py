"""Regression checks for Phimaker's Python-owned diagram bindings."""

import unittest

from phimaker import sixpack, sixpack_from_inclusion, zero_overlap


class PythonBindingsTests(unittest.TestCase):
    inclusion = [(True, 0, []), (False, 0, []), (False, 1, [0, 1])]

    def test_inclusion_diagrams_in_both_modes(self):
        expected = {
            "dom": {0: None},
            "cod": {0: None, 1: 2},
            "im": {0: None},
            "ker": {},
            "cok": {1: 2},
            "rel": {0: None, 1: 2},
        }
        for slow in (False, True):
            with self.subTest(slow=slow):
                diagrams = sixpack_from_inclusion(self.inclusion, num_threads=2, slow=slow)
                for name, values in expected.items():
                    diagram = getattr(diagrams, name)
                    self.assertIsInstance(diagram, dict)
                    self.assertEqual(diagram, values)

    def test_diagram_lookup_iteration_and_copying(self):
        diagrams = sixpack_from_inclusion(self.inclusion, num_threads=2)
        diagram = diagrams.cod
        self.assertEqual(diagram[1], 2)
        self.assertIsNone(diagram[0])
        self.assertEqual(set(diagram), {0, 1})
        self.assertEqual(set(diagram.items()), {(0, None), (1, 2)})
        # A death index is not itself a generator.
        with self.assertRaises(KeyError):
            _ = diagram[2]
        diagram[1] = None
        del diagram[0]
        self.assertEqual(diagram, {1: None})
        # Ensemble getters return independent dictionaries.
        self.assertEqual(diagrams.cod, {0: None, 1: 2})

    def test_identity_chain_map_in_both_modes(self):
        interval = [(0.0, 0, []), (0.0, 0, []), (1.0, 1, [0, 1])]
        for slow in (False, True):
            with self.subTest(slow=slow):
                diagrams, metadata = sixpack(
                    interval, interval, [[0], [1], [2]], num_threads=2, slow=slow
                )
                for name in ("dom", "cod", "im", "ker", "cok"):
                    diagram = getattr(diagrams, name)
                    pairs = sorted(
                        (metadata.times[b], metadata.times[d])
                        for b, d in diagram.items()
                        if d is not None and metadata.times[b] != metadata.times[d]
                    )
                    unpaired = sorted(metadata.times[b] for b, d in diagram.items() if d is None)
                    expected = ([], []) if name in ("ker", "cok") else ([(0.0, 1.0)], [0.0])
                    self.assertEqual((pairs, unpaired), expected)

    def test_zero_overlap(self):
        self.assertEqual(sorted(zero_overlap(self.inclusion)), [(1, 2), (1, 2)])


if __name__ == "__main__":
    unittest.main()
