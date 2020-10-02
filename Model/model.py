import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *


class LSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128, hlayer_num=2, dr_rate=0.2, fr_len=80):
        super(LSTMModel, self).__init__()
        self.emb_fr = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm_fr = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=hlayer_num, dropout=dr_rate,
                               batch_first=True)
        # self.header = nn.Conv1d(in_channels=1, out_channels=vocab_size, stride=hidden_dim, kernel_size=hidden_dim)
        self.header = TimeDist(nn.Linear(hidden_dim, vocab_size), batch_first=True)
        self.fr_len = fr_len

    def forward(self, fr: torch.tensor, fr_len):
        h_fr = self.emb_fr(fr)

        h_fr = torch.nn.utils.rnn.pack_padded_sequence(h_fr, fr_len, batch_first=True, enforce_sorted=False)

        h_fr, _ = self.lstm_fr(h_fr)

        h_fr, _ = torch.nn.utils.rnn.pad_packed_sequence(h_fr, batch_first=True, total_length=self.fr_len)

        # h_fr = h_fr.contiguous().view(h_fr.size(0), 1, -1)

        outputs = self.header(h_fr)

        return outputs


class TimeDist(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDist, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        x_reshape = x.contiguous().view(-1, x.size(-1))

        y = self.module(x_reshape)

        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))

        return y



