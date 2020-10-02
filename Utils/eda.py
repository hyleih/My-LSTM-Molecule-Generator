import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *


if __name__ == "__main__":
    smiles_list = read_smilesset("Data/zinc_250k.smi")
    vocab = ["&", "\n"]
    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        p = parse_smiles(smiles)
        vocab.extend(p)
    vocab = list(set(vocab))

    print(len(vocab))

    with open("Data/vocabulary/zinc_vocab.txt", mode="w") as f:
        for w in vocab:
            f.write(f"{w},")
