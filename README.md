<<<<<<< HEAD
# Diff-prior: Diffusion-based Graph Priors for Structure Discovery

This repository provides **Diff-prior**, a **diffusion-parameterized** correlated structural prior that can be plugged into NRI-family backbones for binary structure recovery on the StructInfer benchmarks. We include three backbones:
- **NRI** (ICML 2018)
- **ACD** (CLeaR 2022)
- **MPM** (AAAI 2021)

> Note: The diffusion prior flags are **not identical** across the three codebases. Please follow the per-backbone commands below.

## Requirements

Common (NRI / ACD):
- python 3.7
- pytorch >= 1.13.1
- numpy, scipy, pandas, tqdm, sklearn, torchinfo
- CUDA 10.0

MPM additionally requires:
- torch-geometric >= 1.3.2  

## Data

Benchmark trajectories are from **StructInfer**:
- Benchmark page: https://structinfer.github.io/benchmark/

You only need to set `--data-path` / `--data_path` to your local dataset directory.

## Running

Key arguments shared by all backbones:
- `--b-network-type`: e.g., `brain_networks`
- `--b-directed`: use directed graphs when applicable
- `--b-simulation-type`: `springs` or `netsims`
- `--b-suffix`: e.g., `15r1` (15 nodes, repetition 1, noise-free)
- `--b-time-steps`: default 49 (if exposed in that backbone)

Below are **single-line** examples for **noise-free Springs trajectories** on **Brain Networks** (15 nodes, rep 1).  
To run **Netsims**, change `--b-simulation-type netsims` (and `--dyn netsims` for MPM) and update the dataset path accordingly.

---
## Run experiments

In general, following args are used to select the trajectories to be used for evaluation:

```python
parser.add_argument('--b-time-steps', type=int, default=49,
                    help='Portion of time series in data to be used in benchmarking. Min = 5, Max = 49')
parser.add_argument('--b-shuffle', action='store_true', default=False,
                    help='Shuffle the data for benchmarking.')
parser.add_argument('--b-network-type', type=str, default='',
                    help='What is the network type of the graph. Please choose from: "brain_networks", "chemical_reaction_networks_in_atmosphere", "food_webs", "gene_coexpression_networks", "gene_regulatory_networks", "intercellular_networks", "landscape_networks", "man-made_organic_reaction_networks", "reaction_networks_inside_living_organism", "social_networks", "vascular_networks".')
parser.add_argument('--b-directed', action='store_true', default=False,
                    help='Default choose trajectories from undirected graphs. Use default only when running experiments on trajectories with gene_coexpression_networks and landscape_networks.')
parser.add_argument('--b-simulation-type', type=str, default='',
                    help='Either "springs" or "netsims".')
parser.add_argument('--b-suffix', type=str, default='',
    help='The rest to locate the exact trajectories. E.g. "50r1_n1" for 50 nodes, rep 1 and noise level 1.'
         ' Or "15r1" for 15 nodes, rep 1 and noise free.')
parser.add_argument('--dyn', type=str, default='',
    help='Type of dynamics. Keep identical to arg "b-simulation-type".')
```

## Example commands 

### (1) NRI + Diff-prior

```bash
python3 train.py \
  --save-probs \
  --b-simulation-type springs \
  --b-network-type brain_networks \
  --b-directed \
  --b-suffix 15r1 \
  --epochs 600 \
  --b-shuffle \
  --save-folder ( the path where you want to save) \
  --data-path ( the path to the dataset) \
  --num-atoms 15 \
  --timesteps 49 \
  --batch-size 64 \
  --lr 0.0005 \
  --encoder-dropout 0.5 \
  --decoder-dropout 0.0 \
  --use-diff-prior \
  --diff-T 100 \
  --lambda-diff 50 \
  --diff-refine-clip 10
```


### (2) ACD + Diff-prior

```bash
python3 train.py \
  --suffix brain_networks \
  --save-probs \
  --b-network-type brain_networks \
  --b-directed \
  --b-simulation-type springs \
  --b-suffix 15r1 \
  --epochs 800 \
  --b-shuffle \
  --save_folder ( the path where you want to save) \
  --data-path ( the path to the dataset) \
  --num_atoms 15 \
  --timesteps 49 \
  --batch_size 64 \
  --encoder mlp \
  --decoder mlp \
  --encoder_dropout 0.5 \
  --decoder_dropout 0.5 \
  --lr 0.0005 \
  --use_diffusion \
  --diff-T 100 \
  --diff-refine-clip 10 \
  --lambda-diff 100
```

### (3) MPM + Diff-prior

```bash
python3 run.py \
  --save-probs \
  --b-network-type brain_networks \
  --b-directed \
  --b-simulation-type springs \
  --b-suffix 15r1 \
  --epochs 800 \
  --save_folder ( the path where you want to save) \
  --data_path ( the path to the dataset) \
  --size 15 \
  --b-time-steps 49 \
  --batch 64 \
  --dyn springs \
  --seed 42 \
  --use-diff-prior \
  --diff-T 100 \
  --diff-refine-clip 10 \
  --lambda-diff 100
```
=======
# Diffprior  
>>>>>>> e6166bdcd9287df9afd41fb0e5b4ac61a10b5ede
