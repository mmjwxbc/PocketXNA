import argparse
import os
import sys

import numpy as np

sys.path.append('.')
from utils.dataset import LMDBDatabase
from utils.data import PocketComplexData, torchify_dict
from utils.nucleic_parser import parse_aptamer_pdb
from utils.parser import PDBProtein


def _select_pocket_block(pdb_path, ligand_pos, radius):
    protein = PDBProtein(str(pdb_path))
    selected = []
    ligand_pos = np.asarray(ligand_pos, dtype=np.float32)
    for residue in protein.residues:
        res_pos = np.asarray([protein.pos[atom_idx] for atom_idx in residue['atoms']], dtype=np.float32)
        if res_pos.size == 0:
            continue
        min_dist = np.linalg.norm(res_pos[:, None, :] - ligand_pos[None, :, :], axis=-1).min()
        if min_dist <= radius:
            selected.append(residue)
    if len(selected) == 0:
        return None
    return protein.residues_to_pdb_block(selected)


def process_one(path, nucleic_type='RNA', pocket_radius=20.0):
    mol_dict = parse_aptamer_pdb(path, nucleic_type=nucleic_type)
    if mol_dict['num_atoms'] == 0:
        raise ValueError('no nucleic atoms parsed')

    pocket_block = _select_pocket_block(path, mol_dict['pos_all_confs'][0].numpy(), pocket_radius)
    if pocket_block is None:
        raise ValueError('no protein pocket residues selected')
    pocket_dict = PDBProtein(pocket_block).to_dict_atom()

    data_id = os.path.splitext(os.path.basename(path))[0]
    record = {}
    record.update({'pocket_' + key: value for key, value in torchify_dict(pocket_dict).items()})
    record.update(mol_dict)
    record.update({
        'data_id': data_id,
        'pdbid': data_id,
        'task': 'aptdesign',
    })
    return PocketComplexData(**record)


def write_assembly(output_dir, data_ids, val_ratio=0.1):
    assembly_dir = os.path.join(output_dir, 'assemblies')
    os.makedirs(assembly_dir, exist_ok=True)
    n_val = max(1, int(len(data_ids) * val_ratio)) if len(data_ids) > 1 else 0
    split_map = {
        'train': data_ids[n_val:],
        'val': data_ids[:n_val],
    }
    for split, split_ids in split_map.items():
        path = os.path.join(assembly_dir, f'split_aptamer_assembly_{split}.lmdb')
        db = LMDBDatabase(path, readonly=False)
        db.add_one('all_dbs', ['aptamer'])
        db.add_one('aptamer', len(split_ids))
        db.add_one('aptdesign-aptamer', len(split_ids))
        db.add_one('datatask_dbs', ['aptdesign-aptamer'])
        for i, data_id in enumerate(split_ids):
            db.add_one(f'aptamer-{i}', data_id)
            db.add_one(f'aptdesign-aptamer-{i}', data_id)
        db.close()
        print(f'wrote {len(split_ids)} {split} assembly entries to {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--nucleic-type', default='RNA')
    parser.add_argument('--pocket-radius', type=float, default=20.0)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    args = parser.parse_args()

    lmdb_dir = os.path.join(args.output_dir, 'aptamer', 'lmdb')
    os.makedirs(lmdb_dir, exist_ok=True)
    db_path = os.path.join(lmdb_dir, 'aptamer.lmdb')
    db = LMDBDatabase(db_path, readonly=False)
    keys = []
    i_record = 0
    failures = []
    for name in sorted(os.listdir(args.input_dir)):
        if not name.lower().endswith('.pdb'):
            continue
        path = os.path.join(args.input_dir, name)
        try:
            record = process_one(path, args.nucleic_type, args.pocket_radius)
            key = record['data_id']
            db.add_one(key, record)
            keys.append(f'{key},{record["data_id"]}\n')
            i_record += 1
        except Exception as exc:
            failures.append(f'{name},{type(exc).__name__},{str(exc)}\n')

    db.close()
    with open(os.path.join(lmdb_dir, 'manifest.csv'), 'w') as f:
        f.write('key,data_id\n')
        f.writelines(keys)
    with open(os.path.join(lmdb_dir, 'failures.csv'), 'w') as f:
        f.write('filename,error_type,error\n')
        f.writelines(failures)
    write_assembly(args.output_dir, [line.split(',', 1)[1].strip() for line in keys], args.val_ratio)
    print(f'wrote {i_record} aptamer records to {db_path}')
    print(f'skipped {len(failures)} pdb files')


if __name__ == '__main__':
    main()
