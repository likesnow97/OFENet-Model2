from pathlib import Path
import pandas as pd

manifest = Path("/nfs/users/lixinglin/projects/ofe_frat_v1/data/Combined/manifest_frames.csv")
df = pd.read_csv(manifest)

print("manifest:", manifest)
print("exists:", manifest.exists())
print("samples:", len(df))
print("columns:", df.columns.tolist())

for col in ["onset_path", "apex_path"]:
    missing = []
    for i, v in enumerate(df[col].astype(str)):
        if not Path(v).exists():
            missing.append((i, v))
    print(f"{col}: missing {len(missing)} / {len(df)}")
    for item in missing[:20]:
        print(item)
