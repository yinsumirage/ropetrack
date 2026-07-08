# Data

Do not commit datasets or generated arrays here.

Expected local links:

```text
data/raw/freihand
data/raw/ho3d_v2
data/raw/ho3d_v3
```

Current rule: keep dataset roots separate and symlink them into `data/raw/`.
Do not force every dataset into one processed layout until a script actually
needs that layout.

```text
data/raw/<dataset>          -> symlink/junction to the real dataset root
data/manifests/<dataset>_*  optional small JSONL/CSV indexes when needed
data/rope/<dataset>_*.jsonl optional rope-distance labels when needed
```

Windows junction example:

```powershell
New-Item -ItemType Junction -Path data\raw\freihand -Target D:\datasets\FreiHAND
New-Item -ItemType Junction -Path data\raw\ho3d_v2 -Target D:\datasets\HO3D_v2
New-Item -ItemType Junction -Path data\raw\ho3d_v3 -Target D:\datasets\HO3D_v3
```

HPC symlink example:

```bash
ln -s /data/wentao/ropetrack/FreiHAND data/raw/freihand
ln -s /data/wentao/ropetrack/HO3D_v2_eval data/raw/ho3d_v2
ln -s /data/wentao/ropetrack/HO3D_v3 data/raw/ho3d_v3
```
