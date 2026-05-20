# Results and Checkpoints

This repository is intended to store source code only.

Training results are generated under:

    results/

Model checkpoints are generated under each run directory by default, but should be moved outside the repository before committing.

Recommended external checkpoint root:

    /nfs/users/lixinglin/projects/checkpoint/microemotion/model2

A run directory usually contains:

    args.json
    subjects.json
    aggregate_metrics.json
    fold_summaries.csv
    fold_predictions/
    fold_<subject>/
    summary_tables/
    checkpoint_location.txt

Do not commit:

    *.pth
    results/
    logs/
    runtime_data/
    generated CSV files
