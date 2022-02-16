import random

import numpy as np
import torch
from tqdm import tqdm

from embeddings import initialize_word2idx


DEFAULT_UNK_INDEX = 1


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_text_vocab(sentences):
    word2idx = initialize_word2idx()
    for s in sentences:
        for w in s:
            if w not in word2idx:
                word2idx[w] = len(word2idx)
    return word2idx


def indexify(sentences, word2idx, unk_index=None):
    def fn(s):
        for w in s:
            if w not in word2idx and unk_index is None:
                raise ValueError
            yield word2idx.get(w, unk_index)
    return [list(fn(s)) for s in tqdm(sentences, desc='indexify')]


