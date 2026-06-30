# Data

Do not commit datasets or generated arrays here.

Expected local links:

```text
data/raw/freihand
data/raw/ho3d_v2
```

Generated files should follow the repo schema:

```text
data/processed/<dataset>/
data/manifests/<dataset>_<split>.jsonl
data/rope/<dataset>_rope.npz
```

Windows junction example:

```powershell
New-Item -ItemType Junction -Path data\raw\freihand -Target D:\datasets\FreiHAND
New-Item -ItemType Junction -Path data\raw\ho3d_v2 -Target D:\datasets\HO3D_v2
```
