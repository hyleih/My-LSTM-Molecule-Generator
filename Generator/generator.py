import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *
from Model.model import LSTMModel


def sample(model, vocab, num_mol, MAX_SEQ=80):
    generated = []
    with torch.no_grad():
        for i in tqdm(range(num_mol)):
            smiles = parse_smiles("&")
            x = np.zeros([1, MAX_SEQ])
            c_path = convert_smiles(smiles, vocab, mode="s2i")
            x[0, :len(c_path)] = c_path
            x = torch.tensor(x, dtype=torch.long)
            x_len = [1]

            while smiles[-1] != "\n" and len(smiles) < MAX_SEQ:
                y = model(x, x_len)
                y = F.softmax(y, dim=2)
                y = y.to('cpu').detach().numpy().copy()
                y = np.array(y[0, len(smiles)-1, :])
                ind = np.random.choice(range(len(y)), p=y)

                x[0, len(smiles)] = ind
                x_len[0] += 1
                smiles.append(vocab[ind])

            generated.append("".join(smiles[1:]).rstrip())

    return generated


if __name__ == "__main__":
    vocab = read_vocabulary("Data/vocabulary/zinc_vocab.txt")
    model = LSTMModel(vocab_size=len(vocab))
    model.load_state_dict(torch.load("Data/model/LSTMModel-zinc.pth"))

    smiles_list = sample(model, vocab, 1000)
    smiles_list = list(set(smiles_list))
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            print(smiles)
