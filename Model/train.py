import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Model.dataset import MolDataset
from Model.model import LSTMModel
from Utils.utils import *


VOCAB_PATH = "Data/vocabulary/zinc_vocab.txt"
SMILES_PATH = "Data/zinc_250k.smi"
# SMILES_PATH = "Data/zinc/zinc_train_parsed20.smi"
VALID_RATE = 0.2
BATCH_SIZE = 1024
SEQ_LEN = 80
EPOCH = 20
INIT = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_LSTMmodel(num_epoch, train_list, test_list, vocab, seq_len, batch_size):
    train_list = [convert_smiles(parse_smiles("&" + s.rstrip("\n") + "\n"), vocab, mode="s2i") for s in train_list]
    test_list = [convert_smiles(parse_smiles("&" + s.rstrip("\n") + "\n"), vocab, mode="s2i") for s in test_list]

    train_dataset = MolDataset(train_list, seq_len)
    valid_dataset = MolDataset(test_list, seq_len)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    model = LSTMModel(vocab_size=len(vocab))
    model.load_state_dict(torch.load("Data/model/LSTMModel-zinc.pth"))

    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    train_loss = []
    valid_loss = []
    for epoch in range(1, num_epoch+1):
        losses = []
        for i, (X, X_len) in enumerate(train_loader):
            optimizer.zero_grad()

            y = model(X.cuda(), X_len.cuda())

            source = y[:, :-1, :]
            source = source.contiguous().view(-1, len(vocab))
            target = X[:, 1:]
            target = target.contiguous().view(-1)

            loss = criterion(source, target.cuda())
            losses.append(float(loss))

            loss.backward()

            optimizer.step()

            print("EPOCH%d:%d, Train loss:%f" % (epoch, i, loss))

        train_loss.append(np.mean(losses))

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
            valid_loss.append(np.mean(valid_losses))

    return model, train_loss, valid_loss


if __name__ == "__main__":
    MODEL = "LSTM"
    train_list = read_smilesset("Data/zinc_250k_iso_train.smi")
    test_list = read_smilesset("Data/zinc_250k_iso_test.smi")
    vocab = read_vocabulary(VOCAB_PATH)
    model, train_loss, valid_loss = train_LSTMmodel(EPOCH, train_list, test_list, vocab, SEQ_LEN, BATCH_SIZE)
    torch.save(model.state_dict(), "Data/model/LSTMModel-zinc.pth")


