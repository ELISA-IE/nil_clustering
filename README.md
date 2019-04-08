# nil_clustering
A repository containing code relevant to our work on nil clustering.

## Requirements
* python==3.6.5
* pytorch==0.4.1
* numpy==1.15.1

## Description
This repository contains two files:

- `model.py` which contains the code for the RNN used to encode name mentions.
- `data.py` which contains the code needed to perform dynamic negative example sampling.

## Details for `model.py`
`model.py` contains a `torch.nn.Module` used to encode name mentions into fixed length
vectors. Please see the PyTorch documentation for details of the behavior of Modules.

The `vocab_size` parameter of the model refers to the number of unique characters used
to represent mentions.

The model expects as input to the forward method a list of `torch.LongTensor`s. Each
tensor should contain a sequence of integers corresponding to the indices of the
characters of the mentions being encoded. The tensors do not need to be padded to be of
the same length. The model will handle this padding internally.

## Details for `data.py`
`data.py` contains a `torch.utils.data.Dataset` which will create the triples for model
training. The `__init__` of the dataset expects a list of pairs of mentions (strings).
Each pair should in the list should refer to the same entity. The `__init__` also
requires a Python `dict` (`char2idx`) which contains a mapping from characters to unique
indices. This mapping is used to construct the `torch.LongTensor`s that represent each
mention. The dataset also has a parameter called `optimized_sampling`. When this flag is
set to `True`, the dataset will provide triples where the negative is selected
dynamically as outlined in the paper. Otherwise, the dataset will select negative
examples for each triple randomly.