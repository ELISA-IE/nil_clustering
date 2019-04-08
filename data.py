import random
from collections import defaultdict
from itertools import chain

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

UNK = "<UNK>"


def vectorize_word(word, char2idx):
    word = [l if l in char2idx else UNK for l in word]
    word = [char2idx[l] for l in word]
    return torch.LongTensor(word)


def vectorize_vocab(vocab, char2idx, device="cpu"):
    return (
        [vectorize_word(word, char2idx).to(device) for word in vocab],
        torch.LongTensor([len(word) for word in vocab]).to(device),
    )


class Data(Dataset):
    def __init__(self, pairings, char2idx, device="cpu", optimized_sampling=True):
        super().__init__()
        self.raw_pairings = pairings
        self.raw_vocab = sorted(set(chain(*pairings)))
        self.vocab_idxs = {word: i for i, word in enumerate(self.raw_vocab)}
        self.char2idx = char2idx
        self.optimized_sampling = optimized_sampling

        self.vocab, self.lengths = vectorize_vocab(
            self.raw_vocab, self.char2idx, device=device
        )
        self.padded_vocab = pad_sequence(self.vocab, batch_first=True)
        self.pairings = [
            (self.vocab_idxs[A], self.vocab_idxs[B]) for A, B in self.raw_pairings
        ]

        # Aggregate information about which mentions are paired with other mentions.
        # These paired mentions will be ineligible as negative samples.
        self.paired = defaultdict(set)
        for A, B in self.pairings:
            self.paired[A].add(B)
            self.paired[B].add(A)

        self.encodings = None
        self.device = device

    def __len__(self):
        return len(self.pairings)

    def __getitem__(self, idx):
        if self.encodings is None:
            raise Exception("Update encodings before calling __getitem__")

        A_idx, P_idx = self.pairings[idx]
        A, P = self.vocab[A_idx], self.vocab[P_idx]
        A_encoded, P_encoded = (
            self.encodings[A_idx].view(1, -1),
            self.encodings[P_idx].view(1, -1),
        )
        AP_dist = F.pairwise_distance(A_encoded, P_encoded).item()

        # Remove ineligible negative examples
        ineligible = sorted(list(self.paired[A_idx]) + list(self.paired[P_idx]))
        mask = torch.ones(len(self.encodings), dtype=torch.uint8)
        mask[ineligible] = 0

        masked_encodings = self.encodings[mask]
        masked_lengths = self.lengths[mask]
        masked_padded_vocab = self.padded_vocab[mask]

        N_dists = F.pairwise_distance(A_encoded, masked_encodings).detach()
        if self.optimized_sampling:
            # Keep only negative examples farther away than the positive sample
            mask = N_dists > AP_dist
            if torch.any(mask):  # If no negative example matches criteria, skip masking
                N_dists = N_dists[mask]
                masked_lengths = masked_lengths[mask]
                masked_padded_vocab = masked_padded_vocab[mask]
            N_idx = torch.argmin(N_dists)
        else:
            N_idx = random.randrange(0, len(masked_padded_vocab))

        N = masked_padded_vocab[N_idx][: masked_lengths[N_idx]]
        return A.to(self.device), P.to(self.device), N.to(self.device)

    def update_encodings(self, model):
        model.eval()
        self.encodings = model(self.vocab).detach()
