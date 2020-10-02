import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *


def evaluate(smiles_list, train_list):
    valids = []
    novels = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valids.append(smiles)
            if smiles not in train_list:
                novels.append(smiles)

    validity = len(valids)/len(smiles_list)
    uniquness = len(list(set(valids)))/len(valids)
    novelty = len(novels)/len(valids)

    return validity, uniquness, novelty

