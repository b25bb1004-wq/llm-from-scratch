# LLM from scratch

My path to building a GPT from the ground up, following Andrej Karpathy's *Neural Networks: Zero to Hero* series. Each folder is one lecture, written out by hand and checked to run top to bottom.

| # | Folder | What it builds | Status |
|---|---|---|---|
| 1 | [`micrograd/`](micrograd/) | A scalar autograd engine and a small MLP, with tests | Done |
| 2 | [`makemore/`](makemore/) | A character-level name generator: bigram model by counting and as a neural net | Part 1 done |

More folders get added as the series goes on (MLP, batch norm, backprop by hand, WaveNet, GPT, tokenizer).

## Setup

```
pip install numpy matplotlib graphviz pytest torch
```

Each folder has its own README with details.
