import numpy as np
import torch
from Bio import PDB

from utils.nucleic_constants import (
    ATOM_ROLE_TO_ID,
    BASE_ATOMS,
    BASE_TO_ID_DNA,
    BASE_TO_ID_RNA,
    PHOSPHATE_ATOMS,
    STANDARD_DNA_RESNAMES,
    STANDARD_RNA_RESNAMES,
    SUGAR_ATOMS,
)


def parse_aptamer_pdb(pdb_path, nucleic_type='RNA'):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure('aptamer', str(pdb_path))

    element = []
    pos = []
    atom_names = []
    res_index = []
    base_type = []
    atom_role = []
    chain_ids = []
    chain_order = []

    if nucleic_type.upper() == 'RNA':
        resname_map = STANDARD_RNA_RESNAMES
        base_to_id = BASE_TO_ID_RNA
    else:
        resname_map = STANDARD_DNA_RESNAMES
        base_to_id = BASE_TO_ID_DNA

    residue_counter = 0

    for model in structure:
        for chain in model:
            local_order = 0
            for residue in chain:
                resname = residue.get_resname().strip()
                if resname not in resname_map:
                    continue

                base = resname_map[resname]
                base_id = base_to_id[base]
                residue_has_atoms = False

                for atom in residue:
                    atom_name = atom.get_name().strip()
                    ele = atom.element.strip()
                    if ele == '' or ele.upper() == 'H':
                        continue

                    element.append(_element_to_atomic_number(ele))
                    pos.append(atom.get_coord())
                    atom_names.append(atom_name)
                    res_index.append(residue_counter)
                    base_type.append(base_id)
                    atom_role.append(infer_atom_role(atom_name))
                    chain_ids.append(chain.id)
                    chain_order.append(local_order)
                    residue_has_atoms = True

                if residue_has_atoms:
                    residue_counter += 1
                    local_order += 1
        break

    pos_array = np.array(pos, dtype=np.float32).reshape(-1, 3)
    return {
        'element': torch.LongTensor(element),
        'pos_all_confs': torch.FloatTensor(pos_array)[None],
        'num_confs': 1,
        'i_conf_list': torch.LongTensor([0]),
        'num_atoms': len(element),
        'bond_index': torch.empty([2, 0], dtype=torch.long),
        'bond_type': torch.empty([0], dtype=torch.long),
        'num_bonds': 0,
        'na_res_index': torch.LongTensor(res_index),
        'na_base_type': torch.LongTensor(base_type),
        'na_atom_role': torch.LongTensor(atom_role),
        'na_atom_name': atom_names,
        'na_chain_id': chain_ids,
        'na_chain_order': torch.LongTensor(chain_order),
        'na_num_residues': residue_counter,
        'na_pair_index': torch.empty([2, 0], dtype=torch.long),
        'na_pair_type': torch.empty([0], dtype=torch.long),
        'na_secondary_mask': torch.zeros([residue_counter], dtype=torch.long),
    }


def infer_atom_role(atom_name):
    if atom_name in PHOSPHATE_ATOMS:
        return ATOM_ROLE_TO_ID['phosphate']
    if atom_name in SUGAR_ATOMS:
        return ATOM_ROLE_TO_ID['sugar']
    if atom_name in BASE_ATOMS:
        return ATOM_ROLE_TO_ID['base']
    return ATOM_ROLE_TO_ID['base']


def _element_to_atomic_number(ele):
    table = {
        'H': 1,
        'C': 6,
        'N': 7,
        'O': 8,
        'P': 15,
        'S': 16,
    }
    ele = ele.upper()
    if ele not in table:
        raise ValueError(f'unsupported nucleic atom element: {ele}')
    return table[ele]
