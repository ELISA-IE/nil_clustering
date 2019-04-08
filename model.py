import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence


def sort_tensors(*args, sort_key=lambda x: x, reverse=True):
    # Bundle so that associated tensors are grouped together into a single tuple. For
    # example, if args consist of a list of tensors, a list of languages for those
    # tensors, and a list of feature vectors, then bundle the first item in each of
    # those groups together. We also add the original index to the bundle so that we can
    # get reorder information if we want it later.
    bundle = [(*ts, i) for i, ts in enumerate(zip(*args))]

    # Sort the bundle so that items move together. We always sort by the first arg
    after_sort = sorted(bundle, key=lambda x: sort_key(x[0]), reverse=reverse)

    # Unpack the sorted bundle
    return zip(*after_sort)


class RNNWordEncoder(nn.Module):
    def __init__(
        self,
        out_dim,
        vocab_size,
        num_layers=1,
        hidden_dim=64,
        embed_dim=32,
        dropout=0.0,
        bidirectional=False,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.dropout_p = dropout
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=self.dropout_p,
        )

        linear_input_dim = hidden_dim if not self.bidirectional else 2 * hidden_dim
        self.relu = nn.ReLU()
        self.linear = nn.Linear(linear_input_dim, out_dim)
        self.dropout = nn.Dropout(self.dropout_p)

    def specs(self):
        return {
            "out_dim": self.out_dim,
            "num_layers": self.num_layers,
            "hidden_dim": self.hidden_dim,
            "embed_dim": self.embed_dim,
            "vocab_size": self.vocab_size,
            "dropout": self.dropout_p,
            "bidirectional": self.bidirectional,
        }

    def forward(self, x):
        reorder = None
        # A packed tensor requires that the tensors be sorted from longest to
        # shortest. We do that here. `reorder` is a tensor we can use
        # to later return the encodings to their proper order.
        x, reorder = sort_tensors(x, sort_key=lambda x: x.size(0))
        lengths = [t.size(0) for t in x]  # a list of len batch_size
        # a list of length batch size of seq_len X hidden_dim
        out = [self.embed(sample) for sample in x]
        out = [self.dropout(sample) for sample in out]
        out = pad_sequence(out)  # seq_len X batch_size X hidden_dim
        out = pack_padded_sequence(out, lengths)  # a packed sequence

        # out has shape n_layers * n_directions X batch_size X hidden_dim
        _, out = self.rnn(out)

        # select the last step of the last layer
        if self.bidirectional:
            # The first dimension contains two outputs from each layer, one for the
            # forward pass and one for the backward pass. We want the forward and the
            # backward outputs from the last layer, so we select the last two items
            # in the first dimension.
            forward, backward = out[-2:, :, :]
            out = torch.cat([forward, backward], dim=1)
        else:  # n_directions is 1, so the final layer output is just the last item
            out = out[-1, :, :]  # batch_size X hidden_dim

        out = self.relu(out)
        out = self.dropout(out)
        out = self.linear(out)  # batch_size X out_dim

        if reorder is not None:
            out = out[np.argsort(reorder), :]

        out = out / torch.norm(out, p=2, dim=1, keepdim=True)  # Spherize data
        return out
