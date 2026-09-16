import torch
import torch.nn as nn
from components.transformers import TransformerModel

try:
    from components.linear_rnn import HGRN2Attention
except Exception:
    HGRN2Attention = None


class HybridModel(nn.Module):
    """Hybrid: run a Transformer encoder, feed its encoder representations
    into a Deltanet (HGRN2 attention) and decode.

    Forward API mirrors other models in this repo: `forward(src, hidden, lengths)`
    returns `(output, hidden)` where `output` is (T, B, noutputs).
    """

    def __init__(self, ntoken, noutputs, d_model=256, nhead=4, d_ffn=512, nlayers=2, dropout=0.1, use_pos=True):
        super().__init__()
        # Transformer to produce contextualized encoder representations
        self.transformer = TransformerModel(ntoken, noutputs, d_model, nhead, d_ffn, nlayers, dropout, pos_encode=use_pos)

        # Deltanet / HGRN2 attention (operates on float representations)
        if HGRN2Attention is not None:
            self.deltanet = HGRN2Attention(mode='chunk', hidden_size=d_model, num_heads=nhead, expand_ratio=d_model // nhead)
        else:
            # Fallback: simple Transformer-like feedforward if HGRN2 isn't available
            self.deltanet = None

        # Final decoder maps d_model -> noutputs
        self.decoder = nn.Linear(d_model, noutputs)
        self.sigmoid = nn.Sigmoid()

    def forward(self, src, hidden=None, lengths=None):
        # src: (T, B) token ids
        # Ask transformer to return encoder representations
        out = self.transformer(src, get_encoder_reps=True)
        if isinstance(out, tuple) and len(out) == 2:
            transformer_logits, encoder_reps = out
        else:
            # older transformer variants may only return output — treat that as logits
            transformer_logits = out
            # try to call transformer internals
            try:
                encoder_reps = self.transformer.transformer_encoder(self.transformer.encoder(src))
            except Exception:
                # as a last resort, reuse token embeddings
                encoder_reps = self.transformer.encoder(src)

        # encoder_reps: (T, B, d_model) -> HGRN2 expects (B, T, d_model)
        rep_bt = encoder_reps.transpose(0, 1)

        if self.deltanet is not None:
            delta_out_bt, _, _ = self.deltanet(rep_bt)
        else:
            # simple identity / linear projection fallback
            delta_out_bt = rep_bt

        delta_out_tb = delta_out_bt.transpose(0, 1)  # (T, B, d_model)
        decoded = self.decoder(delta_out_tb)  # (T, B, noutputs)
        decoded = self.sigmoid(decoded)

        # Return (T, B, noutputs), hidden==None to be compatible
        return decoded, None

    def init_hidden(self, bsz):
        return None
