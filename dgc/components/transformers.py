import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import TransfoXLModel, TransfoXLConfig
from components.self_attention import MultiHeadedAttention
from components.transformer_encoder import Encoder, EncoderLayer, EncoderLayerFFN
from components.position_encodings import (
    PositionalEncoding,
    CosineNpiPositionalEncoding,
    LearnablePositionalEncoding,
)


class TransformerModel(nn.Module):
    """
    PyTorch TransformerEncoder-based model.

    IMPORTANT CHANGE:
      - Returns RAW LOGITS (no sigmoid).
      - Use BCEWithLogitsLoss outside this module for binary classification.
    """

    def __init__(
        self,
        ntoken,
        noutputs,
        d_model,
        nhead,
        d_ffn,
        nlayers,
        dropout=0.5,
        use_embedding=False,
        pos_encode=True,
        bias=False,
        pos_encode_type="absolute",
        max_period=10000.0,
    ):
        super().__init__()
        try:
            from torch.nn import TransformerEncoder, TransformerEncoderLayer
        except Exception:
            raise ImportError("TransformerEncoder module does not exist in PyTorch 1.1 or lower.")

        self.model_type = "Transformer"
        self.src_mask = None

        # positional encoding
        if pos_encode_type in ("absolute", "sin", "sinusoid", "sinus"):
            self.pos_encoder = PositionalEncoding(d_model, dropout, max_period)
        elif pos_encode_type == "cosine_npi":
            self.pos_encoder = CosineNpiPositionalEncoding(d_model, dropout)
        elif pos_encode_type == "learnable":
            self.pos_encoder = LearnablePositionalEncoding(d_model, dropout)
        else:
            raise ValueError(f"Unknown pos_encode_type: {pos_encode_type}")

        self.pos_encode = pos_encode
        self.encoder = nn.Embedding(ntoken, d_model)

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_ffn, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.d_model = d_model
        self.decoder = nn.Linear(d_model, noutputs, bias=bias)
        self.bias = bias

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        # causal mask (lower-triangular)
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        if self.bias:
            self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, has_mask=True, get_attns=False, get_encoder_reps=False):
        """
        src: (T, B) token ids

        Returns:
          logits: (T, B, noutputs)  [RAW logits]
        """
        if has_mask:
            device = src.device
            self.src_mask = self._generate_square_subsequent_mask(len(src)).to(device)
        else:
            self.src_mask = None

        x = self.encoder(src) * math.sqrt(self.d_model)  # (T,B,D)
        if self.pos_encode:
            x = self.pos_encoder(x)

        if get_attns:
            # NOTE: this was previously a best-effort, but may break across PyTorch versions
            attns = []
            encoder_layers = self.transformer_encoder.layers
            inp = x
            for layer in encoder_layers:
                # self_attn returns (attn_output, attn_weights) in many versions
                attn = layer.self_attn(inp, inp, inp, attn_mask=self.src_mask)[1]
                inp = layer(inp, src_mask=self.src_mask)
                attns.append(attn)

        h = self.transformer_encoder(x, self.src_mask)  # (T,B,D)
        logits = self.decoder(h)  # (T,B,noutputs)

        if get_attns:
            return logits, attns
        if get_encoder_reps:
            return logits, h
        return logits


class TransformerXLModel(nn.Module):
    """
    HuggingFace TransfoXL backbone.

    IMPORTANT CHANGE:
      - Returns RAW LOGITS (no sigmoid).
      - Use BCEWithLogitsLoss outside.
    """

    def __init__(self, ntoken, noutputs, d_model, nhead, d_ffn, nlayers, dropout=0.5, use_embedding=False):
        super().__init__()
        self.config = TransfoXLConfig(
            vocab_size=ntoken,
            cutoffs=[],
            d_model=d_model,
            d_embed=d_model,
            n_head=nhead,
            d_inner=d_ffn,
            n_layer=nlayers,
            tie_weights=False,
            d_head=d_model // nhead,
            adaptive=False,
            dropout=dropout,
        )
        self.transformer_encoder = TransfoXLModel(self.config)
        self.decoder = nn.Linear(d_model, noutputs)

    def forward(self, src):
        """
        src: (T, B) token ids
        Returns logits: (T, B, noutputs)
        """
        # HF expects (B,T); it returns (B,T,D)
        h = self.transformer_encoder(src.transpose(0, 1), mems=None)[0]  # (B,T,D)
        h = h.transpose(0, 1)  # (T,B,D)
        logits = self.decoder(h)  # (T,B,noutputs)
        return logits


class SimpleTransformerModel(nn.Module):
    """
    Custom encoder stack.

    IMPORTANT CHANGE:
      - Returns RAW LOGITS (no sigmoid).
      - Use BCEWithLogitsLoss outside.
    """

    def __init__(
        self,
        ntoken,
        noutputs,
        d_model,
        nhead,
        d_ffn,
        nlayers,
        dropout=0.5,
        use_embedding=False,
        pos_encode=True,
        bias=False,
        posffn=False,
        freeze_emb=False,
        freeze_q=False,
        freeze_k=False,
        freeze_v=False,
        freeze_f=False,
        zero_keys=False,
        pos_encode_type="absolute",
        max_period=10000.0,
    ):
        super().__init__()

        self.bias = bias
        self.pos_encode = pos_encode
        self.d_model = d_model

        if self.pos_encode:
            if pos_encode_type in ("absolute", "sin", "sinusoid", "sinus"):
                self.pos_encoder = PositionalEncoding(d_model, dropout, max_period)
            elif pos_encode_type == "cosine_npi":
                self.pos_encoder = CosineNpiPositionalEncoding(d_model, dropout)
            elif pos_encode_type == "learnable":
                self.pos_encoder = LearnablePositionalEncoding(d_model, dropout)
            else:
                raise ValueError(f"Unknown pos_encode_type: {pos_encode_type}")

        self.encoder = nn.Embedding(ntoken, d_model)
        if freeze_emb:
            self.encoder.requires_grad_(False)

        self_attn = MultiHeadedAttention(
            nhead, d_model, dropout, bias, freeze_q, freeze_k, freeze_v, zero_keys
        )

        if not posffn:
            encoder_layers = EncoderLayer(self_attn)
        else:
            feed_forward = nn.Sequential(
                nn.Linear(d_model, d_ffn),
                nn.ReLU(),
                nn.Linear(d_ffn, d_model),
            )
            encoder_layers = EncoderLayerFFN(self_attn, feed_forward)

        self.transformer_encoder = Encoder(encoder_layers, nlayers)

        self.decoder = nn.Linear(d_model, noutputs, bias=bias)
        if freeze_f:
            self.decoder.requires_grad_(False)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        """
        src: (T,B) token ids
        Returns logits: (T,B,noutputs)
        """
        device = src.device
        src_mask = self._generate_square_subsequent_mask(len(src)).to(device)

        x = self.encoder(src) * math.sqrt(self.d_model)  # (T,B,D)
        if self.pos_encode:
            x = self.pos_encoder(x)

        # your custom Encoder expects (B,T,D) + mask (T,T)? you were transposing before,
        # keep the old behavior exactly:
        x_bt = x.transpose(0, 1)  # (B,T,D)
        h_bt = self.transformer_encoder(x_bt, src_mask)  # (B,T,D)
        h = h_bt.transpose(0, 1)  # (T,B,D)

        logits = self.decoder(h)  # (T,B,noutputs)
        return logits
