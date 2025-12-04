import json
import torch
from torch.utils.data import Dataset

class JSONLFeatureDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        self.offsets = self._index_lines()

    def _index_lines(self):
        offsets = []
        offset = 0
        with open(self.file_path, 'r') as f:
            for line in f:
                if line.strip():
                    offsets.append(offset)
                offset += len(line.encode('utf-8'))
        return offsets

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with open(self.file_path, 'r') as f:
            f.seek(self.offsets[idx])
            data = json.loads(f.readline())
            return torch.tensor(data["feature_vector"], dtype=torch.float32)
