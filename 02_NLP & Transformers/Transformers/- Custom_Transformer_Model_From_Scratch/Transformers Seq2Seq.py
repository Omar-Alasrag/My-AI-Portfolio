# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
from transformers import AutoTokenizer, PreTrainedTokenizer, DataCollatorForSeq2Seq
from datasets import Dataset
import os
from tqdm import tqdm
from torchmetrics.text import BLEUScore


MAX_SEQ_LEN = 100

# %%


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len, *args, **kwargs):
        super().__init__(*args, **kwargs)
        pe = torch.zeros((max_seq_len, d_model))
        pos = torch.arange(0, max_seq_len, 1).unsqueeze(1)
        i = torch.arange(0, d_model, 2)
        pe[:, 0::2] = torch.sin(pos / 10000 ** (i / d_model))
        pe[:, 1::2] = torch.cos(pos / 10000 ** (i / d_model))
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(1)]


# en_input
# de_out
class MHA(nn.Module):
    def __init__(self, d_model, n_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.d_model = d_model
        self.q_layer = nn.Linear(d_model, d_model)
        self.k_layer = nn.Linear(d_model, d_model)
        self.v_layer = nn.Linear(d_model, d_model)
        self.attn_fc = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, causal=None, mask=None):
        B, T_out, _ = q.shape
        B, T_in, _ = k.shape
        q = self.q_layer(q).view(B, T_out, self.n_heads, -1).transpose(1, 2)
        k = self.k_layer(k).view(B, T_in, self.n_heads, -1).transpose(1, 2)
        v = self.v_layer(v).view(B, T_in, self.n_heads, -1).transpose(1, 2)

        # (B, n_heads, T_out, -1) * ((B, n_heads, -1, T_in))
        # B, n_heads, T_out * T_in
        logits = q @ k.transpose(-2, -1) / self.d_k**0.5

        if causal is not None:
            causal_mask = torch.tril(torch.ones((T_out, T_in), device=q.device))
            logits = torch.masked_fill(logits, causal_mask == 0, float("-inf"))

        if mask is not None:
            logits = torch.masked_fill(
                logits, mask[:, None, None, :] == 0, float("-inf")
            )

        alphas = torch.softmax(logits, dim=-1)

        # (B, n_heads, T_out, T_in) * (B, n_heads, T_in, _d_model)
        # -> (B, n_heads, T_out, d_model)
        context = alphas @ v

        # (B, n_heads, T_out, d_model)
        context = context.transpose(-2, -3).contiguous().view(B, T_out, self.d_model)

        return self.attn_fc(context)


class FeedForward(nn.Module):
    def __init__(self, d_model, hid_dim, dropout=0.1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main = nn.Sequential(
            nn.Linear(d_model, hid_dim),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, d_model),
        )

    def forward(self, x):
        return self.main(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, hid_dim, dropout=0.1):
        super().__init__()
        self.mha = MHA(d_model, n_heads)
        self.ffn = FeedForward(d_model, hid_dim, dropout=0.1)
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
        self.mha = MHA(d_model, n_heads)
        self.mha_enc_dec = MHA(d_model, n_heads)
        self.ff = FeedForward(d_model, hid_dim, dropout=0.1)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_out, enc_mask=None, dec_mask=None):
        dec_attn = self.mha(x, x, x, causal=True, mask=dec_mask)
        x = self.norm1(x + self.dropout(dec_attn))
        enc_dec_attn = self.mha_enc_dec(x, enc_out, enc_out, mask=enc_mask)
        x = self.norm2(x + self.dropout(enc_dec_attn))
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout(ff_out))
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
        self.embedding = nn.Embedding(output_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_length)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_heads, hid_dim, dropout) for _ in range(n_layers)]
        )

        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_out, enc_mask=None, dec_mask=None):
        x = self.dropout(self.pos_encoding(self.embedding(trg)))
        for layer in self.layers:
            x = layer(x, enc_out, enc_mask, dec_mask)
        return self.fc_out(x)


class Transformer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        d_model,
        n_heads,
        hid_dim,
        n_layers,
        max_length=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = Encoder(
            input_dim, d_model, n_heads, hid_dim, n_layers, max_length, dropout
        )
        self.decoder = Decoder(
            output_dim, d_model, n_heads, hid_dim, n_layers, max_length, dropout
        )

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        enc_out = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_out, src_mask, trg_mask)
        return output


# %%


dset_pd = pd.read_csv("ara.txt", sep="\t")
dset_english = dset_pd.iloc[:, 0].to_numpy().tolist()
dset_arabic = dset_pd.iloc[:, 1].to_numpy().tolist()

dset = Dataset.from_dict({"en": dset_english, "ar": dset_arabic})
# %%

tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
    "Helsinki-NLP/opus-mt-en-ar"
)


tokenizer.add_special_tokens({"bos_token": "<bos>"})
# %%
BOS_ID = tokenizer.bos_token_id


def tokenize_fn(batch):
    encoder_input = tokenizer(batch["en"], truncation=True, max_length=MAX_SEQ_LEN)
    decoder_output = tokenizer(
        text_target=batch["ar"], truncation=True, max_length=MAX_SEQ_LEN
    )
    return {
        "input_ids": encoder_input["input_ids"],
        "attention_mask": encoder_input["attention_mask"],
        "labels": decoder_output["input_ids"],
    }


tokenized_dset = (
    dset.map(tokenize_fn, batched=True)
    .remove_columns(dset.column_names)
    .train_test_split(0.2)
)
# %%


datacollator = DataCollatorForSeq2Seq(tokenizer)

train_loader = DataLoader(
    tokenized_dset["train"], batch_size=40, collate_fn=datacollator, shuffle=True
)

val_loader = DataLoader(
    tokenized_dset["test"], batch_size=40, collate_fn=datacollator, shuffle=False
)


# %%

transformer = Transformer(
    input_dim=len(tokenizer),
    output_dim=len(tokenizer),
    d_model=64,
    n_heads=8,
    hid_dim=128,
    n_layers=2,
    max_length=MAX_SEQ_LEN,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transformer.to(device)
optim = torch.optim.Adam(transformer.parameters(), lr=3e-3)
criterion = nn.CrossEntropyLoss(ignore_index=-100)
metric = BLEUScore().to(device)


# %%
def decode_batch(preds, labels):

    preds = preds.detach().cpu()
    labels = labels.detach().cpu()

    labels = labels.clone()
    labels[labels == -100] = tokenizer.pad_token_id

    pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    label_texts = [[l] for l in label_texts]

    return pred_texts, label_texts


n_epochs = 20
for e in range(n_epochs):
    transformer.train()
    for batch in tqdm(train_loader, leave=False):
        enc_in = batch["input_ids"].to(device)
        enc_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        decoder_input = torch.roll(labels, shifts=1, dims=1)
        decoder_input[:, 0] = BOS_ID
        decoder_input[decoder_input == -100] = tokenizer.pad_token_id

        dec_mask = decoder_input != tokenizer.pad_token_id

        optim.zero_grad()
        decoder_out = transformer(enc_in, decoder_input, enc_mask, dec_mask)
        loss = criterion(decoder_out.view(-1, len(tokenizer)), labels.view(-1))
        loss.backward()
        optim.step()

    print(f"epoch={e} train_loss={loss.item():.4f}")

    if e % 2 == 0:
        transformer.eval()
        metric.reset()
        with torch.no_grad():
            for batch in train_loader:
                enc_in = batch["input_ids"].to(device)
                enc_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                decoder_input = torch.roll(labels, shifts=1, dims=1)
                decoder_input[:, 0] = BOS_ID
                decoder_input[decoder_input == -100] = tokenizer.pad_token_id
                dec_mask = decoder_input != tokenizer.pad_token_id

                decoder_out = transformer(enc_in, decoder_input, enc_mask, dec_mask)
                preds = decoder_out.argmax(dim=-1)
                pred_texts, label_texts = decode_batch(preds, labels)
                metric.update(pred_texts, label_texts)

        bleu_train = metric.compute()
        print(f"epoch {e} training BLEU = {bleu_train:.4f}")

transformer.eval()
metric.reset()

with torch.no_grad():
    for batch in val_loader:

        enc_in = batch["input_ids"].to(device)
        enc_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        decoder_input = torch.roll(labels, shifts=1, dims=1)
        decoder_input[:, 0] = BOS_ID
        decoder_input[decoder_input == -100] = tokenizer.pad_token_id

        dec_mask = decoder_input != tokenizer.pad_token_id

        decoder_out = transformer(enc_in, decoder_input, enc_mask, dec_mask)

        preds = decoder_out.argmax(dim=-1)

        pred_texts, label_texts = decode_batch(preds, labels)

        metric.update(pred_texts, label_texts)

bleu = metric.compute()
print(f"BLEU = {bleu:.4f}")
# %%


def quick_test(sentence):
    transformer.eval()
    inputs = tokenizer(sentence, return_tensors="pt").to(device)
    enc_out = transformer.encoder(inputs["input_ids"])

    decoder_input = torch.tensor([[BOS_ID]], device=device)

    for _ in range(20):
        out = transformer.decoder(decoder_input, enc_out)
        next_token = out[:, -1, :].argmax(-1, keepdim=True)
        decoder_input = torch.cat([decoder_input, next_token], dim=1)

    output_text = tokenizer.decode(decoder_input[0], skip_special_tokens=True)


    print("Input:", sentence)
    print("Output:", output_text.encode("utf-8", errors="replace").decode("utf-8"))


quick_test("he went to it")
