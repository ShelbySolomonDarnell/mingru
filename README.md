# torch-mingru
PyTorch (convolutional) MinGRU implementation based on 

> Feng, Leo, et al. "Were RNNs All We Needed?" (2024).

Convolutional MinGRU based on

> Heindl, Christoph et al. "Convolutional MinGRU" (2024).

## Features
In alignment with torch recurrent modules, **mingru** provides the following core modules
 - `mingru.MinGRUCell` single layer MinGRU
 - `mingru.MinGRU` multi-layer stacked MinGRU 
 - `mingru.MinConv2dGRUCell` single layer convolutional MinGRU
 - `mingru.MinConv2dGRU` multi-layer stacked convolutional MinGRU

Each module supports the following features (if applicable to type)
 - **Parallel**: Efficient log-space parallel evaluation support plus sequential support for testing. Automatically dispatches to the most efficient implementation.
 - **Multilayer**: Stack multiple MinGRU layers via `hidden_sizes=` arguments. When `len(hidden_sizes)>1`, the output hidden states of layer $i$ are passed as inputs to $i+1$. Varying hidden sizes are supported.
 - **Dropout**: Via parameter `dropout=`, when > 0 all inputs of each layer are effected except for the last layer.
 - **Residual**: Residual connections betweeen outputs of minGRU layers via `residual=` argument.
 - **Bias**: Biases in linear layers can be enabled and disabled via the `bias=` argument.
 - **Bidirectional**: Bi-directional processing can be enabled by wrapping RNNs via `mingru.Bidirectional`.
 - **Normalization**: LayerNorm and GroupNorms between stacked MinGRUs via `norm=`argument.
 - **Scripting**: MinGRU is compatible with `torch.jit.script`.
 - **Compatibility**: Interface of `mingru.*` is mostly compatible with that of `torch.nn.GRU/GRUCell`, except that and sequence-first arguments are not supported and bi-directional is provided by `mingru.Bidirectional` wrapper. Cells in **mingru** also support sequence arguments to benefit from parallel computation.

## Installation

```shell
# Install directly from github
pip install git+https://github.com/cheind/mingru.git
```

## Usage

### MinGRU

The following snippet demonstrates a multi-layer stacked MinGRU.

```python
import torch
import mingru

# Instantiate
B, input_size, hidden_sizes, S = 10, 3, [32, 64], 128
rnn = mingru.MinGRU(
    input_size=input_size,
    hidden_sizes=hidden_sizes,
    dropout=0.0,
    residual=True,
).eval()

# Invoke for input x with sequence length S and batch-size B
# This will implicitly assume a 'zero' hidden state
# for each layer.
x = torch.randn(B, S, input_size)
out, h = rnn(x)
assert out.shape == (B, S, 64)
assert h[0].shape == (B, 1, 32)
assert h[1].shape == (B, 1, 64)

# Invoke with initial/previous hidden states.
h = rnn.init_hidden_state(x)
out, h = rnn(torch.randn(B, S, input_size), h=h)

# Sequential prediction pattern
h = rnn.init_hidden_state(x)
out_seq = []
for i in range(x.shape[1]):
    out, h = rnn(x[:, i : i + 1], h=h)
    out_seq.append(out)
out_seq = torch.cat(out_seq, 1)
assert out_seq.shape == (B, S, 64)

# Parallel prediction pattern
out_par, h = rnn(x, rnn.init_hidden_state(x))
assert torch.allclose(out_seq, out_par, atol=1e-4)
```

### MinConv2dGRU

Following sample demonstrates convolutional multi-layer stacked MinGRUs.


```python
import torch
import mingru

B, S = 5, 10
input_size = 3
hidden_sizes = [16, 32, 64]
kernel_sizes = [3, 3, 3]
padding = 1
stride = 2

rnn = mingru.MinConv2dGRU(
    input_size=input_size,
    hidden_sizes=hidden_sizes,
    kernel_sizes=kernel_sizes,
    paddings=padding,
    strides=stride,
    dropout=0.0,
    residual=True,
).eval()

# Invoke for input x with sequence length S and batch-size B
# This will implicitly assume a 'zero' hidden state
# for each layer.
x = torch.randn(B, S, input_size, 64, 64)
out, h = rnn(x)
assert out.shape == (B, S, 64, 8, 8)
assert h[0].shape == (B, 1, 16, 32, 32)
assert h[1].shape == (B, 1, 32, 16, 16)
assert h[2].shape == (B, 1, 64, 8, 8)

# Invoke with initial/previous hidden states.
h = rnn.init_hidden_state(x)
out, h = rnn(x, h=h)

# Sequential prediction pattern
h = rnn.init_hidden_state(x)
out_seq = []
for i in range(x.shape[1]):
    out, h = rnn(x[:, i : i + 1], h=h)
    out_seq.append(out)
out_seq = torch.cat(out_seq, 1)
assert out_seq.shape == (B, S, 64, 8, 8)

# Parallel prediction pattern
out_par, h = rnn(x, rnn.init_hidden_state(x))
assert torch.allclose(out_seq, out_par, atol=1e-4)
```

### Examples

#### Selective Copying
For a more complete example check the [examples/selective_copying.py](./examples/selective_copying.py), which attempts to learn to selectively pick specific tokens in order from a generated sequence.

```shell
python -m examples.selective_copying
    ...
    Step [1941/2000], Loss: 0.0002, Accuracy: 99.61%
    Step [1961/2000], Loss: 0.0002, Accuracy: 100.00%
    Step [1981/2000], Loss: 0.0002, Accuracy: 99.61%
    Validation Accuracy: 100.00%
```

Per default, the example is configured for a small usecase (sequence length 64, vocab size 6, memorize 4), but you might just change to a much larger test by adopting `cfg` dict at the end of the file.

Task is based on
> Gu, Albert, and Tri Dao. "Mamba: Linear-time sequence modeling with selective state spaces." (2023).

#### Video Classification
Trains a video classification network using convolutional MinGRUs from scratch using UCF101 train/test splits. Mimicks the
(first) architecture of 

> Ballas, Nicolas, Li Yao1 Chris Pal, and Aaron Courville. "Delving deeper into convolution networks for learning video representation." (2015).

On fold 1 this achieves a validation top-1 accuracy 95% and 78% on test, which replicates the results from the paper. The architecture uses a VGG16 backbone trained on ImageNet. One can expect better test results when pre-training is done on larger video action datasets.

First, register these environment variables

```shell
# Set path to UCF dataset and annotations
export UCF101_PATH=/path/to/UCF/dir
export UCF101_ANNPATH=/path/to/ann/dir
```

##### Train

```shell
python -m examples.video_classification train -f 1
    ...
    2024-12-01 07:53:26,868: Epoch 7, Step 75961, Loss: 0.0042, Accuracy: 100.00%
    2024-12-01 07:53:43,763: Epoch 7, Step 75981, Loss: 0.1159, Accuracy: 93.75%
    2024-12-01 07:54:05,992: Epoch 7, Step 76000, Validation Accuracy: 99.50%, Validation Loss: 0.00
```

##### Test

Test protocol is based on Paper using 25 clips from each video and perform average/majority voting

```shell
python -m examples.video_classification test -f 1 tmp/video_classifier_best.pt
    ...
    2024-12-01 08:19:27,585: Acc: 0.7048961511382305
    2024-12-01 08:19:27,762: Acc: 0.7047927727099405
    2024-12-01 08:19:27,799: Test accuracy 0.70
```

#### Generative Predictive Text 

Trains and samples from a GPT2-like model, but uses stacked MinGRUs instead of transformers. Adapted from 
[nanoGPT](https://github.com/karpathy/nanoGPT).

##### Train
Dataset is currently restricted to a single text file. We use [Tiny-Shakespeare](https://huggingface.co/datasets/Trelis/tiny-shakespeare)

```shell
python -m examples.nlp train tmp/tinyshakespeare.txt
```

##### Sample
```shell
python -m examples.nlp sample --num-tokens 512 tmp/tinyshakespeare.nlp_best.pt

    ISABELLA:
    One of my sister must confess come,
    And two spain under mine honour humbly out:
    Yea, you'll be made in wicked Pompe. What, ho!
    This is a gallful device shall rise.
    I do beseech you, gentle my lord,
    And bring him well, and nothing but my life,
    But your beauty knows stands with your beauty,
    In your mistress and your brother come.
    ...
```

