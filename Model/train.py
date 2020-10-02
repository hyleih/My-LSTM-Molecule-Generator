import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Model.dataset import MolDataset
from Model.model import LSTMModel
from Utils.utils import *


VOCAB_PATH = "data/vocablary/vocab_zinc_full.txt"
SMILES_PATH = "data/zinc/zinc_scfr_valid.smi"
# SMILES_PATH = "data/zinc/zinc_train_parsed20.smi"
VALID_RATE = 0.2
BATCH_SIZE = 1024
SEQ_LEN = 80
EPOCH = 50
INIT = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_LSTMmodel(num_epoch, smiles_list, vocab, seq_len, batch_size):
    smiles_list = [convert_smiles(parse_smiles("&" + s.rstrip("\n") + "\n"), vocab, mode="s2i") for s in smiles_list]

    sp = int(len(smiles_list) * (1 - VALID_RATE))
    train_dataset = MolDataset(smiles_list[:sp], seq_len)
    valid_dataset = MolDataset(smiles_list[sp:], seq_len)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    model = LSTMModel(vocab_size=len(vocab))
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, num_epoch+1):
        for i, (X, X_len) in enumerate(train_loader):
            optimizer.zero_grad()

            y = model(X.cuda(), X_len.cuda())

            source = y[:, :-1, :]
            source = source.contiguous().view(-1, len(vocab))
            target = X[:, 1:]
            target = target.contiguous().view(-1)

            loss = criterion(source, target.cuda())

            loss.backward()

            optimizer.step()

            print("EPOCH%d:%d, Train loss:%f" % (epoch, i, loss))

        with torch.no_grad():
            valid_losses = []
            for j, (X, X_len) in enumerate(valid_loader):
                pred = model(X.cuda(), X_len.cuda())

                source = pred[:, :-1, :]
                source = source.permute(0, 2, 1)
                target = X[:, 1:]

                loss = criterion(source, target.cuda())

                valid_losses.append(float(loss))

            print("EPOCH%d:, Validation loss:%f" % (epoch, np.mean(valid_losses)))

    return model, smiles_list[:sp], smiles_list[sp:]


if __name__ == "__main__":
    MODEL = "LSTM"
    smiles_list = read_smilesset("Data/zinc_250k.smi")
    vocab = read_vocabulary("Data/vocabulary/zinc_vocab_iso.txt")
    model, train_list, test_list = train_LSTMmodel(EPOCH, smiles_list, vocab, SEQ_LEN, BATCH_SIZE)
    torch.save(model.state_dict(), "Data/model/LSTMModel-zinc.pth")

    with open(f"Data/zinc_train_{MODEL}.smi", mode="w") as f:
        for smiles in smiles_list:
            f.write(f"{smiles}\n")

    with open(f"Data/zinc_test_{MODEL}.smi", mode="w") as f:
        for smiles in smiles_list:
            f.write(f"{smiles}\n")

