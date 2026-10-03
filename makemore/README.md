# makemore from scratch

A character-level language model that generates new names, built while following Andrej Karpathy's "makemore" lectures. It trains on `names.txt` (32,033 names) and learns which character tends to follow which.

## Contents

- `makemore.ipynb` - part 1, the bigram model, done two ways:
  - by counting: a 27x27 table of how often each character follows another, normalised into probabilities and sampled from
  - as a neural net: one-hot inputs, a single 27x27 weight matrix, softmax, and gradient descent on the negative log likelihood
- `names.txt` - the dataset, from Karpathy's makemore repo

## Results

| Model | Loss (avg negative log likelihood) |
|---|---|
| Counting, with `N+1` smoothing | 2.4544 |
| Neural net, 100 steps of gradient descent (lr 50, small weight penalty) | 2.4901 |

Both end up learning the same table, so their samples come out nearly the same. That is the point of part 1: the gradient-based approach reaches what counting gives directly, but it is the one that scales to bigger models.

## Setup

```
pip install torch matplotlib
```

Then open `makemore.ipynb` and run the cells top to bottom.

## Notes

- Karpathy's saved notebook doesn't run top to bottom as is (a few cells use `P` before it's defined, and `W` is left at `(27, 1)`). Those are fixed here.
- His training cell is saved as `for k in range(1)`, because he re-ran it many times during the lecture. This one uses `range(100)` so a single run trains the model.
- `torch.multinomial` gives different samples on this machine than in the lecture even with the same seed. The probabilities match his exactly, so it's a random number generator difference, not a bug.

## Next

Part 2: an MLP over several previous characters.
