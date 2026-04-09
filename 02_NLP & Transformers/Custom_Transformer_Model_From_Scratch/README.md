# 🧠 Custom Transformer Seq2Seq Model (From Scratch)

This project is a "from-scratch" implementation of the **Transformer** architecture based on the seminal paper *"Attention Is All You Need."* It demonstrates a deep mathematical understanding of modern NLP foundations.

## 🏗️ Architecture Features
* **Manual Implementation:** Developed Multi-Head Attention (MHA), Feed-Forward Networks, and Positional Encoding using pure **PyTorch**.
* **Sequence-to-Seqence:** Built both Encoder and Decoder stacks for English-to-Arabic translation.
* **Custom Masking:** Implemented causal masks for the decoder to prevent "cheating" during training and padding masks for variable-length batching.
* **Tokenizer Integration:** Integrated with the `Helsinki-NLP` opus-mt tokenizer for professional-grade subword encoding.

## 📈 Training Metrics
* **Task:** English to Arabic Translation.
* **Evaluation:** Uses **BLEU Score** via `torchmetrics` to validate translation quality.
* **Optimization:** Utilizes Adam optimizer with Cross-Entropy Loss (ignoring padding tokens).

## 🚀 How to Run
1. Ensure `ara.txt` (English-Arabic pairs) is in the root directory.
2. Run `python "Transformers Seq2Seq.py"`.