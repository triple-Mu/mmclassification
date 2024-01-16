import math
import random
from pathlib import Path

import pandas as pd

random.seed(42)

image_root = Path('images')
save_root_train = Path('../data/imagenet/train')
save_root_val = Path('../data/imagenet/val')
df_root = Path('csv文件')
df_label = Path('文件标签汇总数据.csv')
df_label = pd.read_csv(df_label)

for i in df_root.iterdir():
    img_path = image_root / f'image_{i.stem}.png'
    label = df_label.loc[df_label['fileName'] == i.name,
                         'defectType'].values[0]
    sr = save_root_train / label / img_path.name
    if not sr.parent.exists():
        sr.parent.mkdir(parents=True, exist_ok=True)
    sr.write_bytes(img_path.read_bytes())

for lb in save_root_train.iterdir():
    assert lb.is_dir()
    names = [i.name for i in lb.iterdir()]
    random.shuffle(names)
    na = len(names)
    nt = math.ceil(na * 0.9)
    nt = min(na - 1, nt)
    nv = na - nt
    print(f'nt: {nt}\tnv: {nv}')
    assert nv > 0
    for name in names[-nv:]:
        src_path = save_root_train / lb.stem / name
        dst_path = save_root_val / lb.stem / name
        if not dst_path.parent.exists():
            dst_path.parent.mkdir(parents=True, exist_ok=True)
        src_path.rename(dst_path)
