from torch import testing as ptt
import torch
from ocr.attention import LayerNorm
import numpy as np
from ocr.attention import attention, MultiHeadAttention


def hello_test():
    assert 1 == 1
    ptt.assert_allclose(torch.ones(1), torch.ones(1))


def layer_norm_test():
    x = torch.tensor([[1., -1., 2.],
                      [2., 0., 0.],
                      [0., 1., -1.]])

    expect = torch.tensor([[0.2182, -1.0911, 0.8729],
                           [1.1547, -0.5773, -0.5773],
                           [0.0000, 1.0000, -1.0000]])

    layer = LayerNorm(x.shape)

    output = layer(x)
    ptt.assert_allclose(expect, output, rtol=1e-04, atol=0)


def attention_test():
    n_batches = 15
    input_dim = 20
    input_lenght = 10

    x = torch.tensor(np.random.random((n_batches, input_lenght, input_dim)))

    out, attn = attention(x, x, x)

    print(attn)

    assert x.shape == torch.Size([15, 10, 20])
    assert out.shape == torch.Size([15, 10, 20])
    assert attn.shape == torch.Size([15, 10, 10])


def test_multihead_attention():
    h = 8
    d_v = 512
    d_model = 64 * 8

    n_batches = 15
    input_lenght = 10

    multihead_attention = MultiHeadAttention(h, d_model)

    x = torch.from_numpy(np.random.random((n_batches, input_lenght, d_v))).float()

    print(x.shape)

    out = multihead_attention(x, x, x)
    print(out.shape)
    assert out.shape == torch.Size([15, 10, 512])
    assert False
