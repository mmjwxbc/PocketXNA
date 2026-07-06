from utils.sample_noise import DenovoSampleNoiser, register_sample_noise


@register_sample_noise('aptdesign')
class AptDesignSampleNoiser(DenovoSampleNoiser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, task_name='aptdesign')

    def outputs2batch(self, batch, outputs):
        pred_node = outputs['pred_node'].argmax(-1)
        pred_pos = outputs['pred_pos']
        pred_halfedge = outputs['pred_halfedge'].argmax(-1)

        fixed_node = batch['fixed_node'].bool()
        fixed_pos = batch['fixed_pos'].bool()
        fixed_halfedge = batch['fixed_halfedge'].bool()

        batch['node_type'][~fixed_node] = pred_node[~fixed_node]
        batch['node_pos'][~fixed_pos] = pred_pos[~fixed_pos]
        batch['halfedge_type'][~fixed_halfedge] = pred_halfedge[~fixed_halfedge]

        if 'na_backbone_halfedge' in batch:
            idx = batch['na_backbone_halfedge']
            batch['halfedge_type'][idx] = batch['gt_halfedge_type'][idx]

        return batch
