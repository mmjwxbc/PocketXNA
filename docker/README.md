# PocketXMol Docker

This Docker setup recreates the current `pxm_cu128` environment on Ubuntu 22.04
with CUDA 12.8 and PyTorch 2.8.0 CUDA 12.8 wheels.

Build from the repository root:

```bash
docker build -t pocketxmol:pxm_cu128 .
```

Run a quick import check:

```bash
docker run --rm --gpus all pocketxmol:pxm_cu128
```

Start an interactive shell:

```bash
docker run --rm -it --gpus all -v "$PWD":/workspace/PocketXMol pocketxmol:pxm_cu128 bash
```

Notes:

- The conda environment name inside the image is `pxm_cu128`.
- The default command prints the installed PyTorch version and CUDA visibility.
- `utils/requirements.txt` lists `meeko`, `openmm`, and `pdbfixer`, but those
  packages are not installed in the current local `pxm_cu128` environment, so
  they are not included here.

### Aptamer / nucleic-acid training (PocketXNA)

The image also pre-clones [PocketXNA](https://github.com/mmjwxbc/PocketXNA)
into `/opt/PocketXNA`, which adds `featurizer_aptamer`, the `aptdesign`
training config, and the `train_aptdesign.yml` training config.

One-command training launch (single GPU inside the container):

```bash
docker run --rm --gpus all \
    -v /path/to/data_train:/opt/PocketXNA/data_train \
    -v /path/to/lightning_logs:/opt/PocketXNA/lightning_logs \
    pocketxmol:pxm_cu128 \
    /opt/PocketXNA/scripts/train_pl.py \
        --config configs/train/train_aptdesign.yml \
        --num_gpus 1 --device 0 \
        --logdir lightning_logs
```

Or use the bundled launcher (mounted from this repo's `docker/` folder):

```bash
docker run --rm --gpus all \
    -v "$PWD/docker":/workspace/launcher \
    -v /path/to/data_train:/opt/PocketXNA/data_train \
    -v /path/to/lightning_logs:/opt/PocketXNA/lightning_logs \
    pocketxmol:pxm_cu128 \
    bash /workspace/launcher/train_aptdesign.sh \
        --num_gpus 1 --tag myrun
```

Convenience flags handled by `train_aptdesign.sh`:

| flag            | meaning                                                   |
| --------------- | --------------------------------------------------------- |
| `--config FILE` | training config (default: `train_aptdesign.yml`)          |
| `--num_gpus N`  | GPUs to use (default: `nvidia-smi -L` count or `1`)      |
| `--batch_size`  | override `train.batch_size` in the config                 |
| `--max_steps`   | override `train.max_steps`                                |
| `--logdir DIR`  | tensorboard / ckpt root                                    |
| `--tag TAG`     | short suffix for the run name                              |
| `--resume RUN`  | resume from `lightning_logs/RUN/checkpoints/last.ckpt`    |
| `--clone`       | git-clone PocketXNA into `/opt/PocketXNA` if missing      |
