# data_sortedgraph.py
import os
from collections import OrderedDict

import torch
from utils.sentence_processing import sents_to_idx


class SimpleVocab:
    """
    Character-level vocab with the interface expected by utils.sentence_processing.py:
        - get_id(ch)
        - get_word(idx)
        - nwords

    IMPORTANT:
      - <PAD> is a real padding token (id=0)
      - 'T' is EOS (NOT padding)
    """
    def __init__(self, chars):
        self.word2idx = OrderedDict()
        self.idx2word = OrderedDict()

        # 1) PAD must be 0
        self.word2idx["<PAD>"] = 0
        self.idx2word[0] = "<PAD>"

        # 2) Add all characters from data (skip specials if present)
        for ch in chars:
            if ch in ("<PAD>",):
                continue
            if ch not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[ch] = idx
                self.idx2word[idx] = ch

        # 3) Add EOS 'T' if not present
        if "T" not in self.word2idx:
            idx = len(self.word2idx)
            self.word2idx["T"] = idx
            self.idx2word[idx] = "T"

        self.nwords = len(self.word2idx)

    def get_id(self, ch: str) -> int:
        return self.word2idx[ch]

    def get_word(self, idx: int) -> str:
        return self.idx2word[idx]


def build_char_vocab_from_files(texts):
    """
    texts: list[str] (graph encodings)
    Returns SimpleVocab over all characters in texts, plus EOS 'T' and PAD '<PAD>'.
    """
    chars = set()
    for line in texts:
        for ch in line:
            chars.add(ch)
    # make sure EOS exists
    chars.add("T")
    return SimpleVocab(sorted(chars))


class TextCorpus:
    """
    Holds string inputs and '0'/'1' labels.
    """
    def __init__(self, src, tgt):
        self.source = src
        self.target = tgt
        self.noutputs = 1  # scalar label per sequence


class Sampler:
    """
    Batching utility:
      - corpus.source: list[str]
      - corpus.target: list['0' or '1']

    Uses utils.sentence_processing.sents_to_idx for tokenization:
      - appends EOS 'T'
      - pads with <PAD>
    """
    def __init__(self, corpus, voc, batch_size):
        self.voc = voc
        self.batch_size = batch_size
        self.data = corpus.source
        self.targets = corpus.target
        self.noutputs = corpus.noutputs
        self.num_batches = int((len(self.data) + batch_size - 1) / batch_size)

    def __len__(self):
        return len(self.data)

    def get_batch(self, i):
        # Determine batch range
        batch_size = min(self.batch_size, len(self.data) - i)
        word_batch = self.data[i: i + batch_size]
        target_batch = self.targets[i: i + batch_size]

        # Token ids, shape (B, T_max+1) after adding EOS
        batch_ids = sents_to_idx(self.voc, word_batch)

        # Transpose to (T, B)
        source = batch_ids.transpose(0, 1)

        # Lengths = number of non-PAD tokens (includes the single EOS 'T')
        pad_id = self.voc.get_id("<PAD>")
        lengths = (batch_ids != pad_id).sum(dim=1).to(torch.long)

        # Sequence-level labels: (B,1) float32
        labels = torch.tensor(
            [[float(int(l))] for l in target_batch],
            dtype=torch.float32
        )

        return source, labels, lengths


def load_sortedgraph_data(data_root: str, dataset: str = "SortedGraph"):
    """
    Load train + validation bins.

    Expects:
      data_root/dataset/train_src.txt
      data_root/dataset/train_tgt.txt
      data_root/dataset/val_src_bin{i}.txt
      data_root/dataset/val_tgt_bin{i}.txt
    """
    data_dir = os.path.join(data_root, dataset)

    # Train
    train_src_path = os.path.join(data_dir, 'train_src.txt')
    train_tgt_path = os.path.join(data_dir, 'train_tgt.txt')

    with open(train_src_path, 'r') as f:
        train_src = [line.strip() for line in f if line.strip()]
    with open(train_tgt_path, 'r') as f:
        train_tgt = [line.strip() for line in f if line.strip()]

    assert len(train_src) == len(train_tgt), "Train src/tgt length mismatch"
    train_corpus = TextCorpus(train_src, train_tgt)

    # Validation bins
    val_corpora = []
    bin_idx = 0
    while True:
        src_path = os.path.join(data_dir, f'val_src_bin{bin_idx}.txt')
        tgt_path = os.path.join(data_dir, f'val_tgt_bin{bin_idx}.txt')
        if not (os.path.exists(src_path) and os.path.exists(tgt_path)):
            break

        with open(src_path, 'r') as f:
            vsrc = [line.strip() for line in f if line.strip()]
        with open(tgt_path, 'r') as f:
            vtgt = [line.strip() for line in f if line.strip()]

        assert len(vsrc) == len(vtgt), f"Val bin{bin_idx} src/tgt mismatch"
        val_corpora.append(TextCorpus(vsrc, vtgt))
        bin_idx += 1

    return train_corpus, val_corpora, data_dir
