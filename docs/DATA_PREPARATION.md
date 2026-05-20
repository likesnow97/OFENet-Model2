# Data Preparation

This repository does not include raw datasets, generated CSV files, optical-flow arrays, checkpoints, or training logs.

## Expected External Data Layout

The training scripts expect an external dataset root. In our reproduction environment, the root was:

    /nfs/users/lixinglin/data/microemotion

You can replace it with your own path by passing `--raf-path`.

## Combined 3-Class Data

For combined CASME2+SAMM+SMIC 3-class training, the loader expects:

    <raf_path>/3DB/precomputed_tvl1_flows_224/dataset_with_tvl1_flows_224.csv

The standardized columns are:

    Dataset, Subject, Filename, Label, OnsetFrame, ApexFrame, OffsetFrame, OnsetPath, ApexPath, FlowPath

## CASME2 Single-Dataset Data

For CASME2 single-dataset experiments, the loader expects:

    <raf_path>/CASMEII/precomputed_flows_tvl1/dataset_with_flows.csv

## SAMM Single-Dataset Data

For SAMM single-dataset experiments, the loader expects:

    <raf_path>/SAMM/precomputed_flows/dataset_with_flows.csv

The SAMM CSV used in our environment has the following columns:

    Subject, Filename, OnsetFrame, ApexFrame, OffsetFrame, Label_AU, Estimated Emotion, OnsetPath, ApexPath, FlowPath

## Excluded Files

The following files are intentionally excluded from this repository:

    raw images
    precomputed optical-flow arrays
    generated dataset CSV files
    training logs
    experiment results
    model checkpoints
