# Diff-prior: Diffusion-based Graph Priors for Structure Discovery

This repository provides **Diff-prior**, a diffusion-parameterized correlated structural prior for binary structure recovery on the **StructInfer** benchmarks. Diff-prior can be plugged into NRI-family backbones.

We provide implementations based on three representative backbones:

- **NRI**: Neural Relational Inference for Interacting Systems, ICML 2018
- **ACD**: Amortized Causal Discovery: Learning to Infer Causal Graphs from Time-Series Data, CLeaR 2022
- **MPM / NRI-MPM**: Neural Relational Inference with Efficient Message Passing Mechanisms, AAAI 2021

> **Important:** The diffusion-prior flags and several common argument names are not identical across the three codebases. Please follow the per-backbone commands below instead of directly reusing one backbone's command for another backbone.

---

## Requirements

Common requirements for **NRI** and **ACD**:

- Python 3.7
- PyTorch >= 1.13.1
- NumPy
- SciPy
- pandas
- tqdm
- scikit-learn
- torchinfo

Additional requirement for **MPM**:

- torch-geometric >= 1.3.2

The code was tested with CUDA-enabled PyTorch. Please install the PyTorch and CUDA versions that are compatible with your local environment.

---

## Data

Benchmark trajectories are from **StructInfer**:

- Benchmark page: <https://structinfer.github.io/benchmark/>

Please download the corresponding StructInfer trajectories and set the dataset path in the commands below.

Depending on the backbone, the dataset path argument is either:

- `--data-path` for NRI and ACD
- `--data_path` for MPM

---

## Compared Priors

We compare the following edge priors.

### 1. Uniform prior

Equal prior probability is assigned to edge and non-edge classes.

### 2. Fixed sparse prior

The prior probability of an edge is set to `0.03`, and the prior probability of a non-edge is set to `0.97`, following the classical NRI setting.

### 3. Diff-prior

Diff-prior is a learnable, diffusion-parameterized correlated structural prior proposed in our paper.

---

## Common Benchmark Arguments

The following benchmark-related arguments are shared conceptually by the three backbones, although the exact argument names may differ across codebases.

```python
parser.add_argument('--b-time-steps', type=int, default=49,
                    help='Portion of time series in data to be used in benchmarking. Min = 5, Max = 49.')

parser.add_argument('--b-shuffle', action='store_true', default=False,
                    help='Shuffle the data for benchmarking.')

parser.add_argument('--b-network-type', type=str, default='',
                    help='Network type of the graph. Options include: "brain_networks", '
                         '"chemical_reaction_networks_in_atmosphere", "food_webs", '
                         '"gene_coexpression_networks", "gene_regulatory_networks", '
                         '"intercellular_networks", "landscape_networks", '
                         '"man-made_organic_reaction_networks", '
                         '"reaction_networks_inside_living_organism", '
                         '"social_networks", and "vascular_networks".')

parser.add_argument('--b-directed', action='store_true', default=False,
                    help='Use directed graphs when applicable. By default, trajectories from undirected graphs are used.')

parser.add_argument('--b-simulation-type', type=str, default='',
                    help='Simulation type. Choose from "springs" or "netsims".')

parser.add_argument('--b-suffix', type=str, default='',
                    help='Suffix used to locate the exact trajectories. For example, "50r1_n1" means '
                         '50 nodes, repetition 1, and noise level 1. "15r1" means 15 nodes, '
                         'repetition 1, and noise-free trajectories.')

parser.add_argument('--dyn', type=str, default='',
                    help='Type of dynamics. For MPM, keep this identical to --b-simulation-type.')
```

---

## Important Argument Differences

| Item | NRI | ACD | MPM |
|---|---|---|---|
| Training script | `train_diff2.py` | `train.py` | `run.py` |
| Enable Diff-prior | `--use-diff-prior` | `--use_diffusion` | `--use-diff-prior` |
| Data path | `--data-path` | `--data-path` | `--data_path` |
| Save path | `--save-folder` | `--save_folder` | `--save_folder` |
| Number of nodes | `--num-atoms` | `--num_atoms` | `--size` |
| Batch size | `--batch-size` | `--batch_size` | `--batch` |
| Time steps | `--timesteps` | `--timesteps` | `--b-time-steps` |
| Dynamics flag | `--b-simulation-type` | `--b-simulation-type` | `--b-simulation-type` and `--dyn` |

---

## Example Commands

The following examples run **noise-free Springs trajectories** on **Brain Networks** with **15 nodes**, **repetition 1**, and **directed graphs**.

Before running an experiment, set the following paths:

```bash
export DATA_DIR=/path/to/structinfer/dataset
export OUTPUT_DIR=/path/to/save/results
```

To run **Netsims**, change:

```bash
--b-simulation-type netsims
```

For MPM, also change:

```bash
--dyn netsims
```

Please also update `DATA_DIR` to the corresponding Netsims dataset directory.

---

### NRI + Diff-prior

```bash
python3 train_diff2.py \
  --save-probs \
  --b-simulation-type springs \
  --b-network-type brain_networks \
  --b-directed \
  --b-suffix 15r1 \
  --epochs 800 \
  --b-shuffle \
  --save-folder "$OUTPUT_DIR" \
  --data-path "$DATA_DIR" \
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

---

### ACD + Diff-prior

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
  --save_folder "$OUTPUT_DIR" \
  --data-path "$DATA_DIR" \
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

---

### MPM + Diff-prior

```bash
python3 run.py \
  --save-probs \
  --b-network-type brain_networks \
  --b-directed \
  --b-simulation-type springs \
  --b-suffix 15r1 \
  --epochs 800 \
  --save_folder "$OUTPUT_DIR" \
  --data_path "$DATA_DIR" \
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

---

## Running Baselines

The commands above enable **Diff-prior**. To run the uniform-prior or fixed-sparse-prior baselines, use the same dataset and backbone settings, but disable the Diff-prior flag and use the corresponding baseline-prior option implemented in each backbone.

Because the three codebases do not use fully identical argument names, please check the argument parser of each training script before running a baseline:

- NRI: `train_diff2.py`
- ACD: `train.py`
- MPM: `run.py`

---

## Outputs

When `--save-probs` is enabled, the inferred edge probabilities are saved under the specified output directory. The exact output file names and directory structure may differ across NRI, ACD, and MPM because the three implementations are based on different original codebases.

---

## Acknowledgements

This repository builds on NRI-family backbones and adapts or extends parts of the following open-source implementations:

- NRI: <https://github.com/ethanfetaya/NRI>
- ACD: <https://github.com/loeweX/AmortizedCausalDiscovery>
- MPM / NRI-MPM: <https://github.com/hilbert9221/NRI-MPM>

We thank the original authors for releasing their code. If you use this repository, please also cite the corresponding backbone papers.

Before public release, please verify the license terms of the upstream codebases and retain the required copyright and license notices.

---

## Citation

If you find this repository useful, please consider citing our paper.

```bibtex
@misc{diffprior2026,
  title        = {Diff-prior: Diffusion-based Graph Priors for Structure Discovery},
  author       = {Author Names},
  year         = {2026},
  note         = {Please replace this entry with the official citation.}
}
```

Please also consider citing the original backbone papers.

### NRI

```bibtex
@inproceedings{kipf2018neural,
  title     = {Neural Relational Inference for Interacting Systems},
  author    = {Kipf, Thomas and Fetaya, Ethan and Wang, Kuan-Chieh and Welling, Max and Zemel, Richard},
  booktitle = {Proceedings of the 35th International Conference on Machine Learning},
  pages     = {2688--2697},
  year      = {2018},
  volume    = {80},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR}
}
```

### ACD

```bibtex
@article{lowe2022amortized,
  title   = {Amortized Causal Discovery: Learning to Infer Causal Graphs from Time-Series Data},
  author  = {L{\"o}we, Sindy and Madras, David and Zemel, Richard and Welling, Max},
  journal = {Causal Learning and Reasoning},
  year    = {2022}
}
```

### MPM 

```bibtex
@inproceedings{chen2021neural,
  title     = {Neural Relational Inference with Efficient Message Passing Mechanisms},
  author    = {Chen, Siyuan and Wang, Jiahai and Li, Guoqing},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {35},
  number    = {8},
  pages     = {7055--7063},
  year      = {2021}
}
```
