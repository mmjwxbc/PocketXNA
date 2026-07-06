import torch

from utils.nucleic_constants import ATOM_ROLE_TO_ID
from utils.transforms import register_transforms


@register_transforms('featurizer_aptamer')
class FeaturizeAptamer(object):
    def __init__(self, config):
        self.config = config
        atomic_numbers = config.chem.atomic_numbers
        bond_types = config.chem.mol_bond_types

        self.atomic_numbers = torch.LongTensor(atomic_numbers)
        self.mol_bond_types = torch.LongTensor(bond_types)

        self.num_element = self.atomic_numbers.size(0)
        self.num_bond_types = self.mol_bond_types.size(0)
        self.use_mask_node = config.use_mask_node
        self.use_mask_edge = config.use_mask_edge

        self.num_node_types = self.num_element + int(self.use_mask_node)
        self.num_edge_types = self.num_bond_types + 1 + int(self.use_mask_edge)

        self.ele_to_nodetype = {ele: i for i, ele in enumerate(atomic_numbers)}
        self.nodetype_to_ele = {i: ele for i, ele in enumerate(atomic_numbers)}

        self.follow_batch = ['node_type', 'halfedge_type']
        self.exclude_keys = [
            'orig_keys',
            'pos_all_confs',
            'num_confs',
            'i_conf_list',
            'bond_index',
            'bond_type',
            'num_bonds',
            'num_atoms',
            'na_atom_name',
            'na_chain_id',
        ]

    def __call__(self, data):
        data.num_nodes = data.num_atoms
        data.node_type = torch.LongTensor([
            self.ele_to_nodetype[int(ele.item())]
            for ele in data.element
        ])

        atom_pos = data.pos_all_confs[0].float()
        if len(getattr(data, 'pocket_center', [])) > 0:
            atom_pos = atom_pos - data.pocket_center
        else:
            atom_pos = atom_pos - atom_pos.mean(dim=0, keepdim=True)

        data.node_pos = atom_pos
        data.i_conf = data.i_conf_list[0]

        halfedge_index, halfedge_type = self.build_halfedges(data)
        data.halfedge_index = halfedge_index
        data.halfedge_type = halfedge_type

        data.is_peptide = torch.zeros([data.num_nodes], dtype=torch.long)
        data.is_nucleic = torch.ones([data.num_nodes], dtype=torch.long)
        data.ligand_type = 'aptamer'

        data.na_is_phosphate = (data.na_atom_role == ATOM_ROLE_TO_ID['phosphate']).long()
        data.na_is_sugar = (data.na_atom_role == ATOM_ROLE_TO_ID['sugar']).long()
        data.na_is_base = (data.na_atom_role == ATOM_ROLE_TO_ID['base']).long()
        data.na_is_backbone = (data.na_is_phosphate.bool() | data.na_is_sugar.bool()).long()

        return data

    def build_halfedges(self, data):
        edge_type_mat = torch.zeros([data.num_nodes, data.num_nodes], dtype=torch.long)
        for i in range(data.num_bonds * 2):
            src = data.bond_index[0, i]
            dst = data.bond_index[1, i]
            edge_type_mat[src, dst] = data.bond_type[i]

        halfedge_index = torch.triu_indices(data.num_nodes, data.num_nodes, offset=1)
        halfedge_type = edge_type_mat[halfedge_index[0], halfedge_index[1]]
        return halfedge_index, halfedge_type
