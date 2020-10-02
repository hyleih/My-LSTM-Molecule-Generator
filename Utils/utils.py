import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import time

import rdkit.Chem as Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


atoms = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
         "[Sc]", "Ti", "V", "Cr", "[Mn]", "Fe", "[Co]", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
         "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "[Sn]", "Sb", "Te", "I", "Xe"]

atoms_filter = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I", "c", "n", "s", "o", "p", "b"]
sym = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "%", "(", ")", "/", "\n", "&", "=", "#", "\\", "@"]


def convert_smiles(smiles, vocab, mode):
    """
    :param smiles:
    :param vocab: dict of tokens
    :param mode: s2i: string -> int
                 i2s: int -> string
    :return: converted smiles,
    """
    converted = []
    if mode == "s2i":
        for token in smiles:
            converted.append(vocab.index(token))
    elif mode == "i2s":
        for ind in smiles:
            converted.append(vocab[ind])
    return converted


def parse_smiles(smiles):
    parsed = []
    i = 0
    while i < len(smiles):
        asc = ord(smiles[i])
        if 64 < asc < 91:
            if i != len(smiles)-1 and smiles[i:i+2] in atoms:
                parsed.append(smiles[i:i+2])
                i += 2
            else:
                parsed.append(smiles[i])
                i += 1
        elif asc == 91:
            j = i
            while smiles[i] != "]":
                i += 1
            i += 1
            parsed.append(smiles[j:i])

        else:
            parsed.append(smiles[i])
            i += 1

    return parsed


def read_vocabulary(path):
    with open(path) as f:
        vocabulary = []
        s = f.read()
        for w in s.split(","):
            if w is not "":
                vocabulary.append(w)

    return vocabulary


def read_smilesset(path):
    smiles_list = []
    with open(path) as f:
        for smiles in f:
            smiles_list.append(smiles.rstrip())

    return smiles_list


