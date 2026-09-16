# utils/sentence_processing.py
import torch


def pad_seq(seq, max_length, voc):
    """
    Pad with <PAD> up to max_length.
    NOTE: EOS 'T' is appended separately (once).
    """
    pad_id = voc.get_id("<PAD>")
    if len(seq) < max_length:
        seq = seq + [pad_id] * (max_length - len(seq))
    return seq


def sent_to_idx(voc, sent, max_length=-1):
    """
    sent: a string or iterable of characters.
    We map each character to vocab id, append EOS 'T',
    then pad with <PAD> to (max_length+1) if max_length given.
    """
    idx_vec = []
    for ch in sent:
        idx_vec.append(voc.get_id(ch))

    # append EOS
    idx_vec.append(voc.get_id("T"))

    # if max_length provided: pad to max_length+1 (because we added EOS)
    if max_length >= 0:
        idx_vec = pad_seq(idx_vec, max_length + 1, voc)

    return idx_vec


def sents_to_idx(voc, sents):
    """
    sents: list[str]
    Returns tensor of shape (B, T_max+1) (includes EOS),
    padded with <PAD>.
    """
    max_length = max(len(s) for s in sents)  # length without EOS
    all_indexes = [sent_to_idx(voc, sent, max_length) for sent in sents]
    return torch.tensor(all_indexes, dtype=torch.long)


def idx_to_sent(voc, tensor, no_eos=False):
    sent = []
    for idx in tensor:
        ch = voc.get_word(int(idx))
        if no_eos and ch == "T":
            continue
        sent.append(ch)
    return sent


def idx_to_sents(voc, tensors, no_eos=False):
    """
    tensors: (T,B) or (B,T) depending on caller.
    Here we assume (T,B) and transpose to iterate batch.
    """
    tensors = tensors.transpose(0, 1)
    return [idx_to_sent(voc, t, no_eos=no_eos) for t in tensors]
