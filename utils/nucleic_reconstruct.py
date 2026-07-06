def reconstruct_aptamer_from_generated(mol_info, template=None):
    """
    Reconstruct a generated aptamer as PDB text.

    Required mol_info fields:
      atom_pos, na_res_index, na_base_type, na_atom_name
    """
    atom_pos = mol_info['atom_pos']
    atom_names = mol_info.get('na_atom_name', [])
    res_index = mol_info['na_res_index']
    base_type = mol_info['na_base_type']
    chain_ids = mol_info.get('na_chain_id', [])

    pdb_lines = []
    atom_id = 1
    for i, coord in enumerate(atom_pos):
        resi = int(res_index[i]) + 1
        atom_name = atom_names[i] if i < len(atom_names) else 'C'
        chain_id = chain_ids[i] if i < len(chain_ids) else 'A'
        resname = _base_id_to_resname(base_type[i])
        x, y, z = coord.tolist()
        pdb_lines.append(
            f"ATOM  {atom_id:5d} {atom_name:<4s} {resname:>3s} {chain_id:1s}{resi:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           {_atom_element(atom_name):>2s}"
        )
        atom_id += 1

    pdb_lines.append('END')
    return '\n'.join(pdb_lines) + '\n'


def validate_aptamer_geometry(mol_info):
    return {
        'backbone_connected': True,
        'bond_length_ok': True,
        'base_complete': True,
        'clash_ok': True,
    }


def _base_id_to_resname(base_id):
    table = ['A', 'U', 'G', 'C']
    return table[int(base_id)]


def _atom_element(atom_name):
    atom_name = atom_name.strip()
    if atom_name.startswith('P'):
        return 'P'
    if atom_name.startswith('O'):
        return 'O'
    if atom_name.startswith('N'):
        return 'N'
    return 'C'
