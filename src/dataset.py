"""dataset.py
Utilities to load local parallel corpus (train.en / train.vi / valid.en / valid.vi)
and produce a HuggingFace Dataset ready for training.
"""
from datasets import Dataset, DatasetDict
import os

def load_parallel(data_dir: str, split_name: str = 'train'):
    en_path = os.path.join(data_dir, f"{split_name}.en")
    vi_path = os.path.join(data_dir, f"{split_name}.vi")
    if not os.path.exists(en_path) or not os.path.exists(vi_path):
        raise FileNotFoundError(f"Missing files for split {split_name}: {en_path} or {vi_path}")
    with open(en_path, 'r', encoding='utf-8') as f:
        en_lines = [l.strip() for l in f if l.strip()]
    with open(vi_path, 'r', encoding='utf-8') as f:
        vi_lines = [l.strip() for l in f if l.strip()]
    if len(en_lines) != len(vi_lines):
        raise ValueError(f"Mismatch in number of lines: en={len(en_lines)} vs vi={len(vi_lines)}")
    data = [{'en': e, 'vi': v} for e,v in zip(en_lines, vi_lines)]
    return Dataset.from_list(data)

def prepare_dataset(data_dir: str = 'data'):
    datasets = {}
    datasets['train'] = load_parallel(data_dir, 'train')
    valid_en = os.path.join(data_dir, 'valid.en')
    valid_vi = os.path.join(data_dir, 'valid.vi')
    if os.path.exists(valid_en) and os.path.exists(valid_vi):
        datasets['validation'] = load_parallel(data_dir, 'valid')
    return DatasetDict(datasets)

if __name__ == '__main__':
    ds = prepare_dataset('data')
    for k in ds:
        print(k, len(ds[k]))
