import textwrap
import tempfile
import unittest
from pathlib import Path

import torch
from easydict import EasyDict
from torch_geometric.data import Batch

from utils.data import PocketComplexData
from utils.nucleic_parser import parse_aptamer_pdb
from utils.transforms import get_transforms
from utils.sample_noise import get_sample_noiser


def _write_minimal_rna_pdb(path):
    path.write_text(
        textwrap.dedent(
            """\
            ATOM      1  P     A A   1       0.000   0.000   0.000  1.00  0.00           P
            ATOM      2  O5'   A A   1       1.500   0.000   0.000  1.00  0.00           O
            ATOM      3  C1'   A A   1       2.000   1.000   0.000  1.00  0.00           C
            ATOM      4  N9    A A   1       2.500   1.500   0.000  1.00  0.00           N
            ATOM      5  C4    A A   1       3.000   2.000   0.000  1.00  0.00           C
            TER
            END
            """
        )
    )


def _aptamer_featurizer_cfg():
    return EasyDict(
        name="featurizer_aptamer",
        nucleic_type="RNA",
        use_mask_node=True,
        use_mask_edge=True,
        chem=EasyDict(
            atomic_numbers=[6, 7, 8, 15],
            mol_bond_types=[1, 2, 3, 4],
        ),
    )


class AptamerBackendTest(unittest.TestCase):
    def test_legacy_featurizers_are_config_registered(self):
        pocket = get_transforms(EasyDict(name="featurizer_pocket", knn=4))
        mol = get_transforms(
            EasyDict(
                name="featurizer_mol",
                use_mask_node=True,
                use_mask_edge=True,
                chem=EasyDict(atomic_numbers=[6, 7, 8], mol_bond_types=[1, 2, 3]),
            )
        )

        self.assertGreater(pocket.feature_dim, 0)
        self.assertEqual(mol.num_node_types, 4)


    def test_parse_aptamer_pdb_outputs_nucleic_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "rna.pdb"
            _write_minimal_rna_pdb(pdb_path)

            parsed = parse_aptamer_pdb(pdb_path, nucleic_type="RNA")

        self.assertEqual(parsed["element"].tolist(), [15, 8, 6, 7, 6])
        self.assertEqual(tuple(parsed["pos_all_confs"].shape), (1, 5, 3))
        self.assertEqual(parsed["na_num_residues"], 1)
        self.assertEqual(parsed["na_base_type"].tolist(), [0, 0, 0, 0, 0])
        self.assertEqual(len(parsed["na_atom_name"]), 5)


    def test_pocket_complex_batch_increments_aptamer_indices(self):
        data1 = PocketComplexData(
            node_type=torch.zeros(3, dtype=torch.long),
            halfedge_type=torch.zeros(3, dtype=torch.long),
            halfedge_index=torch.tensor([[0, 0, 1], [1, 2, 2]]),
            na_pair_index=torch.tensor([[0], [2]]),
            na_res_pair_index=torch.tensor([[0], [1]]),
            na_num_residues=2,
        )
        data2 = PocketComplexData(
            node_type=torch.zeros(2, dtype=torch.long),
            halfedge_type=torch.zeros(1, dtype=torch.long),
            halfedge_index=torch.tensor([[0], [1]]),
            na_pair_index=torch.tensor([[0], [1]]),
            na_res_pair_index=torch.tensor([[0], [0]]),
            na_num_residues=1,
        )

        batch = Batch.from_data_list([data1, data2])

        self.assertEqual(batch.na_pair_index[:, 1].tolist(), [3, 4])
        self.assertEqual(batch.na_res_pair_index[:, 1].tolist(), [2, 2])


    def test_aptamer_featurizer_adds_graph_and_nucleic_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "rna.pdb"
            _write_minimal_rna_pdb(pdb_path)
            parsed = parse_aptamer_pdb(pdb_path, nucleic_type="RNA")
        parsed.update(
            {
                "num_atoms": len(parsed["element"]),
                "num_bonds": 1,
                "bond_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                "bond_type": torch.tensor([1, 1], dtype=torch.long),
            }
        )
        data = PocketComplexData.from_pocket_mol_dicts(mol_dict=parsed, task="aptdesign")

        featurizer = get_transforms(_aptamer_featurizer_cfg())
        data = featurizer(data)

        self.assertEqual(tuple(data.node_type.shape), (5,))
        self.assertEqual(tuple(data.halfedge_index.shape), (2, 10))
        self.assertEqual(data.halfedge_type.sum().item(), 1)
        self.assertEqual(data.is_nucleic.tolist(), [1, 1, 1, 1, 1])
        self.assertEqual(data.na_is_phosphate.sum().item(), 2)
        self.assertEqual(data.na_is_sugar.sum().item(), 1)
        self.assertEqual(data.na_is_base.sum().item(), 2)


    def test_aptdesign_transform_sets_fixed_masks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "rna.pdb"
            _write_minimal_rna_pdb(pdb_path)
            parsed = parse_aptamer_pdb(pdb_path, nucleic_type="RNA")
        parsed.update(
            {
                "num_atoms": len(parsed["element"]),
                "num_bonds": 1,
                "bond_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                "bond_type": torch.tensor([1, 1], dtype=torch.long),
            }
        )
        data = PocketComplexData.from_pocket_mol_dicts(mol_dict=parsed, task="aptdesign")
        data = get_transforms(_aptamer_featurizer_cfg())(data)

        transform = get_transforms(
            EasyDict(
                name="aptdesign",
                settings=EasyDict(mode=EasyDict(fixed_motif=1), structure=EasyDict(free=1)),
                fixed_motif=EasyDict(residue_index=[0]),
            ),
            mode="train",
        )
        data = transform(data)

        self.assertTrue(set(["fixed_node", "fixed_pos", "fixed_halfedge", "fixed_halfdist"]).issubset(data.keys()))
        self.assertEqual(data.node_fixed_motif.numel(), 5)
        self.assertEqual(data.fixed_node.sum().item(), 5)
        self.assertEqual(data.node_backbone.numel(), 3)
        self.assertEqual(data.node_base.numel(), 2)


    def test_aptdesign_noiser_preserves_fixed_outputs(self):
        cfg = EasyDict(
            name="aptdesign",
            num_steps=100,
            prior=EasyDict(
                node=EasyDict(name="categorical", prior_type="predefined", prior_probs=[1, 1, 1]),
                pos=EasyDict(name="allpos", pos=EasyDict(name="gaussian_simple", sigma_max=1)),
                edge=EasyDict(name="categorical", prior_type="tomask_half"),
            ),
            level=EasyDict(name="uniform", min=0.0, max=1.0),
        )
        noiser = get_sample_noiser(cfg, 3, 3, mode="sample", device="cpu")
        batch = {
            "node_type": torch.tensor([0, 1]),
            "node_pos": torch.zeros(2, 3),
            "halfedge_type": torch.tensor([1]),
            "fixed_node": torch.tensor([1, 0]),
            "fixed_pos": torch.tensor([1, 0]),
            "fixed_halfedge": torch.tensor([1]),
            "gt_halfedge_type": torch.tensor([1]),
        }
        outputs = {
            "pred_node": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
            "pred_pos": torch.ones(2, 3),
            "pred_halfedge": torch.tensor([[0.0, 0.0, 1.0]]),
        }

        updated = noiser.outputs2batch(batch, outputs)

        self.assertEqual(updated["node_type"].tolist(), [0, 2])
        self.assertEqual(updated["node_pos"][0].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(updated["node_pos"][1].tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(updated["halfedge_type"].tolist(), [1])


if __name__ == "__main__":
    unittest.main()
