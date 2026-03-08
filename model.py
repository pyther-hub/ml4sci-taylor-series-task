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
# 3.  LSTM SEQ2SEQ WITH ATTENTION
# ══════════════════════════════════════════════════════════════════════════════

class BahdanauAttention(nn.Module):
    """Additive (Bahdanau) attention over encoder outputs."""

    def __init__(self, enc_dim: int, dec_dim: int):
        super().__init__()
        self.W_enc = nn.Linear(enc_dim, dec_dim, bias=False)
        self.W_dec = nn.Linear(dec_dim, dec_dim, bias=False)
        self.v     = nn.Linear(dec_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: torch.Tensor,   # (B, dec_dim)
        encoder_outputs: torch.Tensor,  # (B, S, enc_dim)
        mask: torch.Tensor,             # (B, S) True where PAD
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        context : (B, enc_dim)
        weights : (B, S)
        """
        # (B, S, dec_dim) + (B, 1, dec_dim)
        energy = torch.tanh(
            self.W_enc(encoder_outputs) + self.W_dec(decoder_hidden).unsqueeze(1)
        )
        scores = self.v(energy).squeeze(-1)            # (B, S)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)        # (B, S)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, enc_dim)
        return context, weights


class CoeffPredLSTM(nn.Module):
    """LSTM Seq2Seq with Bahdanau attention for Taylor coefficient prediction.

    Public interface matches CoeffPredTransformer:
      forward(src, tgt)            -> logits (B, tgt_len-1, VOCAB_SIZE)
      generate_batch(src, max_len) -> List[List[int]]
      generate(src, max_len)       -> List[int]

    Parameters
    ----------
    d_model            : embedding dimension
    hidden_size        : LSTM hidden size
    num_encoder_layers : encoder LSTM layers
    num_decoder_layers : decoder LSTM layers
    dropout            : dropout rate
    max_seq_len        : unused, kept for config consistency
    """

    def __init__(
        self,
        d_model:            int   = 256,
        hidden_size:        int   = 256,
        num_encoder_layers: int   = 2,
        num_decoder_layers: int   = 2,
        dropout:            float = 0.1,
        max_seq_len:        int   = 512,
    ):
        super().__init__()
        self.d_model     = d_model
        self.hidden_size = hidden_size
        self.num_decoder_layers = num_decoder_layers

        # Shared embedding
        self.embedding = nn.Embedding(VOCAB_SIZE, d_model, padding_idx=PAD_ID)

        # Bidirectional encoder
        self.encoder = nn.LSTM(
            d_model, hidden_size, num_encoder_layers,
            batch_first=True, bidirectional=True, dropout=dropout if num_encoder_layers > 1 else 0,
        )
        enc_out_dim = hidden_size * 2  # bidirectional

        # Project encoder final states to decoder initial states
        self.h_proj = nn.Linear(hidden_size * 2, hidden_size)
        self.c_proj = nn.Linear(hidden_size * 2, hidden_size)

        # Attention
        self.attention = BahdanauAttention(enc_out_dim, hidden_size)

        # Decoder LSTM: input = embedding + context
        self.decoder = nn.LSTM(
            d_model + enc_out_dim, hidden_size, num_decoder_layers,
            batch_first=True, dropout=dropout if num_decoder_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)

        # Output projection (weight-tied to embedding)
        self.fc_out = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.fc_out.weight = self.embedding.weight

        # Project decoder output to d_model if hidden_size != d_model
        if hidden_size != d_model:
            self.out_proj = nn.Linear(hidden_size, d_model, bias=False)
        else:
            self.out_proj = None

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, mean=0.0, std=self.d_model ** -0.5)
        with torch.no_grad():
            self.embedding.weight[PAD_ID].fill_(0)

    def _encode(self, src: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple]:
        """Encode source.

        Returns
        -------
        enc_outputs  : (B, S, 2*hidden)
        src_pad_mask : (B, S) True where PAD
        dec_init     : (h0, c0) each (num_decoder_layers, B, hidden)
        """
        src_pad_mask = src == PAD_ID
        emb = self.dropout(self.embedding(src))  # (B, S, d_model)

        # Pack for efficiency
        lengths = (~src_pad_mask).sum(dim=1).cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False,
        )
        enc_out, (h_n, c_n) = self.encoder(packed)
        enc_out, _ = nn.utils.rnn.pad_packed_sequence(enc_out, batch_first=True)  # (B, S, 2*H)

        # h_n: (2*num_enc_layers, B, H) — concat forward/backward for each layer
        # Take the top layer's forward and backward hidden states
        num_enc_layers = h_n.shape[0] // 2
        h_fwd = h_n[2 * num_enc_layers - 2]  # top layer forward
        h_bwd = h_n[2 * num_enc_layers - 1]  # top layer backward
        c_fwd = c_n[2 * num_enc_layers - 2]
        c_bwd = c_n[2 * num_enc_layers - 1]

        h_cat = torch.cat([h_fwd, h_bwd], dim=-1)  # (B, 2*H)
        c_cat = torch.cat([c_fwd, c_bwd], dim=-1)

        h0 = torch.tanh(self.h_proj(h_cat))  # (B, H)
        c0 = torch.tanh(self.c_proj(c_cat))

        # Expand for all decoder layers
        h0 = h0.unsqueeze(0).expand(self.num_decoder_layers, -1, -1).contiguous()
        c0 = c0.unsqueeze(0).expand(self.num_decoder_layers, -1, -1).contiguous()

        return enc_out, src_pad_mask, (h0, c0)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Teacher-forced forward pass.

        Parameters
        ----------
        src : (B, src_len)
        tgt : (B, tgt_len)

        Returns
        -------
        logits : (B, tgt_len-1, VOCAB_SIZE)
        """
        tgt_in = tgt[:, :-1]  # (B, T-1)
        B, T = tgt_in.shape

        enc_out, src_pad_mask, (h, c) = self._encode(src)
        tgt_emb = self.dropout(self.embedding(tgt_in))  # (B, T, d_model)

        outputs = []
        for t in range(T):
            # Attention context from top decoder hidden layer
            context, _ = self.attention(h[-1], enc_out, src_pad_mask)  # (B, 2*H)
            # Decoder input: concat embedding + context
            dec_input = torch.cat([tgt_emb[:, t:t+1, :],
                                   context.unsqueeze(1)], dim=-1)  # (B, 1, d_model+2*H)
            dec_out, (h, c) = self.decoder(dec_input, (h, c))  # (B, 1, H)
            outputs.append(dec_out)

        out = torch.cat(outputs, dim=1)  # (B, T, H)
        if self.out_proj is not None:
            out = self.out_proj(out)
        return self.fc_out(out)  # (B, T, V)

    @torch.inference_mode()
    def generate_batch(
        self,
        src: torch.Tensor,
        max_len: int = 256,
    ) -> List[List[int]]:
        """Greedy decode a batch.

        Parameters
        ----------
        src     : (B, src_len)
        max_len : maximum tokens to generate

        Returns
        -------
        List of length B — each element is the predicted token ID list.
        """
        device = src.device
        B = src.shape[0]

        enc_out, src_pad_mask, (h, c) = self._encode(src)

        cur_token = torch.full((B, 1), SOS_ID, dtype=torch.long, device=device)
        finished  = torch.zeros(B, dtype=torch.bool, device=device)
        outputs   = [[] for _ in range(B)]

        for _ in range(max_len):
            emb = self.embedding(cur_token)  # (B, 1, d_model)
            context, _ = self.attention(h[-1], enc_out, src_pad_mask)
            dec_input = torch.cat([emb, context.unsqueeze(1)], dim=-1)
            dec_out, (h, c) = self.decoder(dec_input, (h, c))  # (B, 1, H)

            proj = dec_out
            if self.out_proj is not None:
                proj = self.out_proj(proj)
            logits = self.fc_out(proj[:, 0, :])  # (B, V)
            next_ids = logits.argmax(dim=-1)      # (B,)
            next_ids = next_ids.masked_fill(finished, PAD_ID)

            for i in range(B):
                if not finished[i]:
                    tid = next_ids[i].item()
                    outputs[i].append(tid)
                    if tid == EOS_ID:
                        finished[i] = True

            cur_token = next_ids.unsqueeze(1)
            if finished.all():
                break

        return outputs

    @torch.inference_mode()
    def generate(
        self,
        src: torch.Tensor,
        max_len: int = 256,
    ) -> List[int]:
        """Greedy decode a single sample."""
        return self.generate_batch(src, max_len=max_len)[0]


# ══════════════════════════════════════════════════════════════════════════════
# 4.  SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cpu")
    B, S, T = 4, 12, 10

    def _make_dummy_data():
        src = torch.randint(1, VOCAB_SIZE, (B, S)).to(device)
        tgt = torch.randint(1, VOCAB_SIZE, (B, T)).to(device)
        src[:, 0] = SOS_ID; src[:, -1] = EOS_ID
        tgt[:, 0] = SOS_ID; tgt[:, -1] = EOS_ID
        return src, tgt

    def _test_model(model, name):
        print(f"\n{'=' * 50}")
        print(f"  Smoke test: {name}")
        print(f"{'=' * 50}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"VOCAB_SIZE     : {VOCAB_SIZE}")
        print(f"Trainable params: {total_params:,}")

        src, tgt = _make_dummy_data()

        logits = model(src, tgt)
        assert logits.shape == (B, T - 1, VOCAB_SIZE), f"Got {logits.shape}"
        print(f"forward logits : {logits.shape}  ✓")

        model.eval()
        results = model.generate_batch(src, max_len=20)
        assert len(results) == B
        print(f"generate_batch : {len(results)} sequences  ✓")
        for i, r in enumerate(results):
            print(f"  [{i}] {r[:8]}{'...' if len(r) > 8 else ''}")

        ids = model.generate(src[:1], max_len=20)
        print(f"generate (1)   : {ids[:8]}  ✓")
        print(f"  {name} smoke test passed.")

    # Test Transformer
    _test_model(
        CoeffPredTransformer(
            d_model=64, nhead=4,
            num_encoder_layers=2, num_decoder_layers=2,
            dim_feedforward=128, dropout=0.1,
        ).to(device),
        "CoeffPredTransformer",
    )

    # Test LSTM
    _test_model(
        CoeffPredLSTM(
            d_model=64, hidden_size=64,
            num_encoder_layers=2, num_decoder_layers=2,
            dropout=0.1,
        ).to(device),
        "CoeffPredLSTM",
    )

    print("\nAll smoke tests passed.")
