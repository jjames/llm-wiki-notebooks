# LLM Wiki Notebooks

Jupyter notebooks accompanying the [llm-wiki](../llm-wiki) knowledge base — a structured curriculum for studying ML foundations through frontier AI research.

## Notebooks

| # | Notebook | Topics |
|---|----------|--------|
| 01 | `01_perceptron.ipynb` | Perceptron, sigmoid, decision boundaries |
| 02 | `02_mlp_backprop.ipynb` | MLP, backpropagation, vanishing gradients |
| 03 | `03_cnn.ipynb` | Convolutional networks, feature maps |
| 04 | `04_rnn.ipynb` | Recurrent networks, hidden state |
| 05 | `05_lstm.ipynb` | LSTM, gating, long-range dependencies |
| 06 | `06_seq2seq.ipynb` | Sequence-to-sequence, encoder-decoder |
| 07 | `07_attention.ipynb` | Attention mechanism, Bahdanau attention |
| 08 | `08_transformer.ipynb` | Full transformer architecture |
| 09 | `09_gpt.ipynb` | Autoregressive language modeling, GPT |
| 10 | `10_bert.ipynb` | Masked language modeling, BERT |
| 11 | `11_superposition.ipynb` | Toy models of superposition (Elhage et al. 2022) |
| 12 | `12_sparse_autoencoder.ipynb` | Sparse autoencoders, monosemanticity (Bricken et al. 2023) |

## Setup

```bash
pip install torch numpy matplotlib jupyter
jupyter notebook
```

Notebooks 11–12 implement experiments from the mechanistic interpretability literature and are best read alongside the corresponding wiki pages on [[superposition]] and [[towards-monosemanticity]].
