import argparse
import os
import sys

import torch
import yaml
from easydict import EasyDict

sys.path.append('.')
from utils.data import PocketComplexData
from utils.nucleic_parser import parse_aptamer_pdb
from utils.nucleic_reconstruct import reconstruct_aptamer_from_generated
from utils.transforms import Compose, get_transforms


def load_config(path):
    with open(path, 'r') as f:
        return EasyDict(yaml.safe_load(f))


def get_input_data(input_aptamer, nucleic_type='RNA', pocmol_args=None):
    aptamer_dict = parse_aptamer_pdb(input_aptamer, nucleic_type=nucleic_type)
    return PocketComplexData.from_pocket_mol_dicts(
        pocket_dict=None,
        mol_dict=aptamer_dict,
        task='aptdesign',
        **(pocmol_args or {}),
    )


def build_transform(config):
    transforms = []
    if 'featurizer_pocket' in config.transforms:
        transforms.append(get_transforms(config.transforms.featurizer_pocket))
    transforms.append(get_transforms(config.transforms.featurizer))
    transforms.append(get_transforms(config.task.transform, mode='use'))
    return Compose(transforms)


def write_template_reconstruction(data, outdir):
    os.makedirs(outdir, exist_ok=True)
    mol_info = {
        'atom_pos': data.gt_node_pos if 'gt_node_pos' in data else data.node_pos,
        'na_res_index': data.na_res_index,
        'na_base_type': data.na_base_type,
        'na_atom_name': data.na_atom_name,
        'na_chain_id': data.na_chain_id,
    }
    pdb = reconstruct_aptamer_from_generated(mol_info)
    out_path = os.path.join(outdir, 'aptamer.pdb')
    with open(out_path, 'w') as f:
        f.write(pdb)
    with open(os.path.join(outdir, 'gen_info.csv'), 'w') as f:
        f.write('filename,tag\naptamer.pdb,template_reconstruct\n')
    with open(os.path.join(outdir, 'log.txt'), 'w') as f:
        f.write('sample_aptamer scaffold path executed; diffusion loop is not enabled in this minimal backend.\n')
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_task', required=True)
    parser.add_argument('--config_model', required=False)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    config = load_config(args.config_task)
    torch.manual_seed(config.sample.get('seed', 2024))
    data_cfg = config.data
    data = get_input_data(
        input_aptamer=data_cfg.input_aptamer,
        nucleic_type=data_cfg.get('nucleic_type', 'RNA'),
        pocmol_args=data_cfg.get('pocmol_args', {}),
    )
    data = build_transform(config)(data)
    out_path = write_template_reconstruction(data, args.outdir)
    print(f'wrote {out_path}')


if __name__ == '__main__':
    main()
