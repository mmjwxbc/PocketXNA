import numpy as np
import torch

from utils.transforms import register_transforms


APT_SETTINGS = {
    'mode': ['denovo', 'optimize', 'fixed_motif', 'dock'],
    'structure': ['free', 'secondary_fixed', 'motif_fixed'],
}


@register_transforms('aptdesign')
class AptDesignTransform:
    def __init__(self, config, **kwargs):
        self.config = config
        self.mode = kwargs.get('mode', 'test')
        self.settings = config.get('settings', {})
        self.fixed_motif = config.get('fixed_motif', None)
        self.exclude_keys = ['na_atom_name', 'na_chain_id']

    def __call__(self, data):
        setting = self.sample_setting()
        data['task_setting'] = setting

        data = self.make_groups(data, setting)
        data = self.set_fixed(data, setting)

        if self.mode in ['test', 'use', 'sample']:
            data = self.prepare_sample(data, setting)

        return data

    def sample_setting(self):
        output = {}
        for key, value_dict in self.settings.items():
            options = list(value_dict.keys())
            weights = np.array(list(value_dict.values()), dtype=float)
            weights = weights / weights.sum()
            output[key] = np.random.choice(options, p=weights)
        if 'mode' not in output:
            output['mode'] = 'denovo'
        return output

    def make_groups(self, data, setting):
        data['node_backbone'] = torch.nonzero(
            data.na_is_phosphate.bool() | data.na_is_sugar.bool()
        )[:, 0]
        data['node_phosphate'] = torch.nonzero(data.na_is_phosphate.bool())[:, 0]
        data['node_sugar'] = torch.nonzero(data.na_is_sugar.bool())[:, 0]
        data['node_base'] = torch.nonzero(data.na_is_base.bool())[:, 0]

        if self.fixed_motif is not None:
            fixed_res = torch.LongTensor(self.fixed_motif.get('residue_index', []))
            if fixed_res.numel() > 0:
                is_fixed_res = (data.na_res_index[:, None] == fixed_res[None]).any(dim=1)
                data['node_fixed_motif'] = torch.nonzero(is_fixed_res)[:, 0]
            else:
                data['node_fixed_motif'] = torch.empty([0], dtype=torch.long)
        else:
            data['node_fixed_motif'] = torch.empty([0], dtype=torch.long)

        return data

    def set_fixed(self, data, setting):
        n_node = data.node_type.shape[0]
        n_halfedge = data.halfedge_type.shape[0]

        fixed_node = torch.zeros(n_node, dtype=torch.long)
        fixed_pos = torch.zeros(n_node, dtype=torch.long)
        fixed_halfedge = torch.zeros(n_halfedge, dtype=torch.long)
        fixed_halfdist = torch.zeros(n_halfedge, dtype=torch.long)

        mode = setting.get('mode', 'denovo')
        if mode == 'dock':
            fixed_node[:] = 1
            fixed_halfedge[:] = 1
        elif mode == 'optimize':
            fixed_halfedge[:] = 1
        elif mode == 'fixed_motif':
            motif = data['node_fixed_motif']
            fixed_node[motif] = 1
            fixed_pos[motif] = 1

        data['fixed_node'] = fixed_node
        data['fixed_pos'] = fixed_pos
        data['fixed_halfedge'] = fixed_halfedge
        data['fixed_halfdist'] = fixed_halfdist
        return data

    def prepare_sample(self, data, setting):
        data['gt_node_type'] = data.node_type.clone()
        data['gt_node_pos'] = data.node_pos.clone()
        data['gt_halfedge_type'] = data.halfedge_type.clone()

        mode = setting.get('mode', 'denovo')
        if mode == 'denovo':
            data.node_type[:] = 0
            data.node_pos[:] = 0
            data.halfedge_type[:] = 0
        elif mode == 'fixed_motif':
            motif = data['node_fixed_motif']
            movable = torch.ones(data.num_nodes, dtype=torch.bool)
            movable[motif] = False
            data.node_type[movable] = 0
            if motif.numel() > 0:
                data.node_pos[movable] = data.node_pos[motif].mean(dim=0, keepdim=True)

        return data
