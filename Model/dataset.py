import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *


class MolDataset(torch.utils.data.Dataset):
    def __init__(self, smiles_list, seq_len):
        super(MolDataset, self).__init__()
        self.smiles_list = smiles_list
        self.seq_len = seq_len

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, item):
        x = self.smiles_list[item]
        x_len = len(x)
        x = torch.tensor(x, dtype=torch.long)
        buf = torch.zeros(self.seq_len, dtype=torch.long)
        buf[:len(x)] = x
        x = buf
        # x = torch.nn.functional.one_hot(x.to(torch.int64), num_classes=len(self.vocab))

        return x, x_len
