import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=1024):
        super().__init__()
        pos = torch.arange(0, max_length).unsqueeze(1)
        i = torch.arange(0, d_model // 2) * 2

        pe = torch.zeros(max_length, d_model)

        divisor = 10000 ** (i / d_model)
        pe[:, 0::2] = torch.sin(pos / divisor)  # T*1     d_model/2
        pe[:, 1::2] = torch.cos(pos / divisor)  # T*1     d_model/2
        self.register_buffer("pe", pe)
        # B * T * d_model

    def forward(self, x):
        return x + self.pe[: x.shape[1], :].unsqueeze(0)  # (1, T, d_model)


class MHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model

        self.key = nn.Linear(d_model, d_model)
        self.query = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

        self.attn_fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None, causal=None):
        B, T_q, _ = q.shape
        _, T_k, _ = k.shape

        # q -> (B, h, T_q, d_k)
        q = self.query(q).view(B, T_q, self.n_heads, self.d_k).transpose(1, 2)
        # k v -> (B, h, T_k, d_k)
        k = self.key(k).view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)
        v = self.value(v).view(B, T_k, self.n_heads, self.d_k).transpose(1, 2)

        # (B, h, T_q, d_k) * (B, h, d_k, T_k) -> (B, h, T_q, T_k)
        scores = (
            q
            @ k.transpose(-1, -2)
            / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        )

        if mask is not None:
            scores = scores.masked_fill(mask[:, None, None, :] == 0, float("-inf"))

        if causal:
            causal_mask = torch.tril(torch.ones((T_q, T_q), device=scores.device))
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)

        # (B, h, T_q, T_k) @ (B, h, T_k, d_k) -> (B, h, T_q, d_k)
        output = attn_weights @ v

        output = output.transpose(1, 2).contiguous().view(B, T_q, self.d_model)

        return self.attn_fc(output)


class FeedForward(nn.Module):
    def __init__(self, d_model, hid_dim):
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(d_model, hid_dim),
            nn.ReLU(True),
            nn.Linear(hid_dim, d_model),
        )

    def forward(self, x):
        return self.main(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, hid_dim, dropout=0.1):
        super().__init__()
        self.mha = MHA(d_model, n_heads)
        self.ffn = FeedForward(d_model, hid_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.mha(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        n_heads,
        hid_dim,
        n_layers,
        max_length=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.tok_embedding = nn.Embedding(input_dim, d_model)
        self.pos_embedding = PositionalEncoding(d_model, max_length)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, hid_dim, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        x = self.dropout(self.pos_embedding(self.tok_embedding(src)))
        for layer in self.layers:
            x = layer(x, mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, hid_dim, dropout=0.1):
        super().__init__()
        self.self_mha = MHA(d_model, n_heads)
        self.enc_dec_mha = MHA(d_model, n_heads)
        self.ffn = FeedForward(d_model, hid_dim)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask=None, trg_mask=None):
        # attention with causal masking
        self_attn = self.self_mha(x, x, x, mask=trg_mask, causal=True)
        x = self.norm1(x + self.dropout(self_attn))

        # encoder-decoder attention
        enc_dec_attn = self.enc_dec_mha(x, enc_out, enc_out, mask=src_mask)
        x = self.norm2(x + self.dropout(enc_dec_attn))

        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_out))
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim,
        d_model,
        n_heads,
        hid_dim,
        n_layers,
        max_length=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.tok_embedding = nn.Embedding(output_dim, d_model)
        self.pos_embedding = PositionalEncoding(d_model, max_length)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, hid_dim, dropout) for _ in range(n_layers)]
        )
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_out, src_mask=None, trg_mask=None):
        x = self.dropout(self.pos_embedding(self.tok_embedding(trg)))
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, trg_mask)
        return self.fc_out(x)  # (B, T, output_dim)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        trg_vocab_size,
        d_model=128,
        n_heads=8,
        hid_dim=512,
        n_layers=2,
        max_length=100,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size, d_model, n_heads, hid_dim, n_layers, max_length, dropout
        )
        self.decoder = Decoder(
            trg_vocab_size, d_model, n_heads, hid_dim, n_layers, max_length, dropout
        )

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        enc_out = self.encoder(src, mask=src_mask)
        out = self.decoder(trg, enc_out, src_mask, trg_mask)
        return out
