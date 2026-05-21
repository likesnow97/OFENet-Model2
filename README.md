# OFE-FRAT-Model2

This repository provides a cleaned and maintainable implementation of Model2 for micro-expression recognition.

Model2 is based on `ImprovedResNetV22`, which uses precomputed onset-apex TV-L1 optical flow as input. The model contains:

- ResNet18 backbone
- RFR: Region Feature Reweighting
- ADAF: Adaptive Dual Attention Fusion
- LOSO training protocol
- 3-class and 5-class experiment entries
- ablation variants

## Repository Contents

    model2/
    process/
    configs/
    scripts/
    tools/
    tests/
    train_model2_combined3.py
    train_model2_single5.py
    summarize_model2_results.py
    metrics2.py
    utils.py
    confusion_matrix.py
    docs/

## External Dependency

This project depends on the following external repository:

- https://github.com/MabelLeeeee/gazeEmotion

`gazeEmotion` is declared in `requirements.txt` as a pip-installable Git dependency and is also included as a Git submodule for source-level traceability.

### Clone with submodules

```bash
git clone --recurse-submodules https://github.com/likesnow97/OFENet-Model2.git
cd OFENet-Model2
````

If the repository has already been cloned without submodules, run:

```bash
git submodule update --init --recursive
```

### Install dependencies

```bash
pip install -r requirements.txt
```



## What Is Not Included

This repository does not include:

    raw datasets
    generated CSV files
    precomputed optical-flow arrays
    checkpoints
    training logs
    result tables
    runtime_data

These files are excluded to keep the repository lightweight.

## Installation

Create the environment with conda:

    conda env create -f environment.yml
    conda activate ofe_frat

Or install packages manually:

    pip install -r requirements.txt

## Data Preparation

Prepare the external dataset directory and precomputed optical-flow files before training.

See:

    docs/DATA_PREPARATION.md

## Model2 Mixed 3-Class Training

Example:

    cd /path/to/OFE-FRAT-Model2

    python train_model2_combined3.py \
      --raf-path /path/to/microemotion \
      --subjects all \
      --num-classes 3 \
      --epochs 100 \
      --batch-size 32 \
      --optimizer adam \
      --lr 1e-4 \
      --device cuda \
      --variant full

## Model2 Single-Dataset 5-Class Training

CASME2 example:

    python train_model2_single5.py \
      --db casme2 \
      --raf-path /path/to/microemotion \
      --subjects all \
      --num-classes 5 \
      --epochs 100 \
      --batch-size 32 \
      --device cuda \
      --variant full

SAMM example:

    python train_model2_single5.py \
      --db samm \
      --raf-path /path/to/microemotion \
      --subjects all \
      --num-classes 5 \
      --epochs 100 \
      --batch-size 32 \
      --device cuda \
      --variant full

## Ablation Variants

Available variants:

    full
    no_pretrain
    no_rfr
    no_adaf
    no_rfr_no_adaf

Example:

    python train_model2_combined3.py \
      --raf-path /path/to/microemotion \
      --subjects all \
      --num-classes 3 \
      --epochs 100 \
      --batch-size 32 \
      --device cuda \
      --variant no_rfr



## Summarizing Results

After training, summarize a run with:

    python summarize_model2_results.py \
      --run-dir results/<run_name> \
      --source-csv /path/to/source.csv

## Checkpoints

Checkpoints should be stored outside this repository.

Recommended checkpoint root:

    /path/to/checkpoint/microemotion/model2

See:

    docs/RESULTS_AND_CHECKPOINTS.md

## License

See LICENSE.
