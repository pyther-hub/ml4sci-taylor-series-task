"""
coeff_pred_model.py
===================
Seq2Seq Transformer for the Taylor coefficient prediction task.

Uses the fixed 42-token vocabulary from coeff_pred_dataset.py — both the
encoder (source: function prefix) and decoder (target: coefficient prefix)
share the same vocabulary and the same embedding table.

Public interface
----------------
  forward(src, tgt)              -> logits (B, tgt_len-1, VOCAB_SIZE)
  generate_batch(src, max_len)   -> List[List[int]]   greedy, batched
  generate(src, max_len)         -> List[int]          greedy, single sample
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn

from dataset import EOS_ID, PAD_ID, SOS_ID, VOCAB_SIZE


# ══════════════════════════════════════════════════════════════════════════════
# 1.  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MODEL
# ══════════════════════════════════════════════════════════════════════════════

class CoeffPredTransformer(nn.Module):
    """Seq2Seq Transformer for predicting one Taylor coefficient prefix.

    Encoder input  : function f(x) in prefix notation
    Decoder output : Taylor coefficient c_n(a) in prefix notation

    Shared vocabulary
    -----------------
    Both src and tgt use the same 42-token fixed vocabulary from
    coeff_pred_dataset.py.  Encoder and decoder share a single embedding table,
    which reduces the parameter count and leverages the fact that the same
    mathematical tokens appear on both sides.

    Parameters
    ----------
    d_model            : embedding / model dimension (must be divisible by nhead)
    nhead              : number of attention heads
    num_encoder_layers : depth of the encoder stack
    num_decoder_layers : depth of the decoder stack
    dim_feedforward    : inner dimension of the FFN sub-layer
    dropout            : dropout rate applied throughout
    max_seq_len        : maximum sequence length (positional encoding table size)
    """

    def __init__(
        self,
        d_model:            int   = 128,
        nhead:              int   = 4,
        num_encoder_layers: int   = 3,
        num_decoder_layers: int   = 3,
        dim_feedforward:    int   = 256,
        dropout:            float = 0.1,
        max_seq_len:        int   = 1024,
    ):
        super().__init__()
        assert d_model % nhead == 0, (
            f"d_model ({d_model}) must be divisible by nhead ({nhead})"
        )

        self.d_model = d_model

        # Shared embedding for src and tgt (same fixed vocabulary)
        self.embedding    = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_ID)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_seq_len)

        # Encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_encoder_layers, enable_nested_tensor=False
        )

        # Decoder
        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        # Output projection (tied to the shared embedding weight)
        self.fc_out = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.fc_out.weight = self.embedding.weight   # weight tying

        self._init_weights()

    # ─────────────────────────────────────────────────────────────────────────
    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.d_model ** -0.5)
        with torch.no_grad():
            self.embedding.weight[PAD_ID].fill_(0)

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _causal_mask(sz: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular bool mask that blocks attention to future positions."""
        return torch.triu(
            torch.ones(sz, sz, dtype=torch.bool, device=device), diagonal=1
        )

    # ─────────────────────────────────────────────────────────────────────────
    def _encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the source sequence.

        Parameters
        ----------
        src : (B, src_len)

        Returns
        -------
        memory          : (B, src_len, d_model)
        src_pad_mask    : (B, src_len)  True where PAD
        """
        src_pad_mask = src == PAD_ID                                   # (B, S)
        emb          = self.embedding(src) * math.sqrt(self.d_model)  # (B, S, D)
        emb          = self.pos_encoding(emb)
        memory       = self.encoder(emb, src_key_padding_mask=src_pad_mask)
        return memory, src_pad_mask

    # ─────────────────────────────────────────────────────────────────────────
    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward pass.

        Parameters
        ----------
        src : (B, src_len)  — source token IDs  [SOS fn_tokens EOS]
        tgt : (B, tgt_len)  — target token IDs  [SOS coeff_tokens EOS]

        Returns
        -------
        logits : (B, tgt_len-1, VOCAB_SIZE)
            Predicts positions 1 … tgt_len-1 (i.e. teacher-forces on tgt[:,:-1]).
        """
        tgt_in  = tgt[:, :-1]                                         # (B, T-1)
        tgt_len = tgt_in.shape[1]

        memory, src_pad_mask = self._encode(src)

        tgt_causal_mask = self._causal_mask(tgt_len, src.device)      # (T-1, T-1)
        tgt_pad_mask    = tgt_in == PAD_ID                            # (B, T-1)

        tgt_emb = self.embedding(tgt_in) * math.sqrt(self.d_model)   # (B, T-1, D)
        tgt_emb = self.pos_encoding(tgt_emb)

        out = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
        )                                                              # (B, T-1, D)

        return self.fc_out(out)                                        # (B, T-1, V)

    # ─────────────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def generate_batch(
        self,
        src:     torch.Tensor,
        max_len: int = 256,
    ) -> List[List[int]]:
        """Greedy decode a whole batch with KV-cached self-attention.

        Each step processes only the single new token (O(T) per step) instead
        of re-running the full sequence through the decoder (O(T²) total).

        Parameters
        ----------
        src     : (B, src_len)
        max_len : maximum tokens to generate (not counting SOS)

        Returns
        -------
        List of length B — each element is the predicted token ID list
        (SOS stripped, EOS included if generated, PAD not included).
        """
        device = src.device
        B      = src.shape[0]

        memory, src_pad_mask = self._encode(src)          # (B, S, D), (B, S)

        # KV cache: for each decoder layer, accumulate the normed hidden states
        # that serve as keys and values in self-attention.  At step t the cache
        # holds norm1(x) for positions 0 … t, so new tokens only need Q.
        sa_kv_cache: List[torch.Tensor] = [
            torch.empty(B, 0, self.d_model, device=device)
            for _ in self.decoder.layers
        ]

        cur_token = torch.full((B, 1), SOS_ID, dtype=torch.long, device=device)
        finished  = torch.zeros(B, dtype=torch.bool, device=device)
        outputs   = [[] for _ in range(B)]

        for step in range(max_len):
            # Embed only the current token; apply PE at position `step`.
            x = self.embedding(cur_token) * math.sqrt(self.d_model)  # (B, 1, D)
            x = x + self.pos_encoding.pe[:, step : step + 1]         # (B, 1, D)

            new_sa_kv_cache: List[torch.Tensor] = []
            for l_idx, layer in enumerate(self.decoder.layers):
                # ── Self-attention (KV cache) ──────────────────────────────
                normed_sa = layer.norm1(x)                            # (B, 1, D)
                kv_seq    = torch.cat(
                    [sa_kv_cache[l_idx], normed_sa], dim=1
                )                                                     # (B, T+1, D)
                new_sa_kv_cache.append(kv_seq)

                # Q = current token only; K = V = all positions so far.
                # No causal mask needed: each step naturally sees only past tokens.
                sa_out, _ = layer.self_attn(
                    normed_sa, kv_seq, kv_seq, need_weights=False,
                )                                                     # (B, 1, D)
                x = x + layer.dropout1(sa_out)

                # ── Cross-attention ────────────────────────────────────────
                normed_ca = layer.norm2(x)
                ca_out, _ = layer.multihead_attn(
                    normed_ca, memory, memory,
                    key_padding_mask=src_pad_mask,
                    need_weights=False,
                )                                                     # (B, 1, D)
                x = x + layer.dropout2(ca_out)

                # ── Feed-forward ───────────────────────────────────────────
                normed_ff = layer.norm3(x)
                ff_out    = layer.linear2(
                    layer.dropout(layer.activation(layer.linear1(normed_ff)))
                )
                x = x + layer.dropout3(ff_out)

            sa_kv_cache = new_sa_kv_cache

            logits   = self.fc_out(x[:, 0, :])           # (B, V)
            next_ids = logits.argmax(dim=-1)              # (B,)
            next_ids = next_ids.masked_fill(finished, PAD_ID)

            for i in range(B):
                if not finished[i]:
                    tid = next_ids[i].item()
                    outputs[i].append(tid)
                    if tid == EOS_ID:
                        finished[i] = True

            cur_token = next_ids.unsqueeze(1)             # (B, 1)
            if finished.all():
                break

        return outputs

    # ─────────────────────────────────────────────────────────────────────────
    @torch.inference_mode()
    def generate(
        self,
        src:     torch.Tensor,
        max_len: int = 256,
    ) -> List[int]:
        """Greedy decode a single sample.

        Parameters
        ----------
        src     : (1, src_len)
        max_len : maximum tokens to generate

        Returns
        -------
        List[int] — generated token IDs (SOS stripped, EOS included if hit)
        """
        return self.generate_batch(src, max_len=max_len)[0]


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cpu")

    model = CoeffPredTransformer(
        d_model=64, nhead=4,
        num_encoder_layers=2, num_decoder_layers=2,
        dim_feedforward=128, dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"VOCAB_SIZE   : {VOCAB_SIZE}")
    print(f"Trainable params: {total_params:,}")

    B, S, T = 4, 12, 10
    src = torch.randint(1, VOCAB_SIZE, (B, S)).to(device)
    tgt = torch.randint(1, VOCAB_SIZE, (B, T)).to(device)
    src[:, 0] = SOS_ID; src[:, -1] = EOS_ID
    tgt[:, 0] = SOS_ID; tgt[:, -1] = EOS_ID

    # Forward
    logits = model(src, tgt)
    assert logits.shape == (B, T - 1, VOCAB_SIZE), f"Got {logits.shape}"
    print(f"forward logits : {logits.shape}  ✓")

    # generate_batch
    model.eval()
    results = model.generate_batch(src, max_len=20)
    assert len(results) == B
    print(f"generate_batch : {len(results)} sequences  ✓")
    for i, r in enumerate(results):
        print(f"  [{i}] {r[:8]}{'...' if len(r) > 8 else ''}")

    # generate (single)
    ids = model.generate(src[:1], max_len=20)
    print(f"generate (1)   : {ids[:8]}  ✓")

    print("\nSmoke test passed.")
