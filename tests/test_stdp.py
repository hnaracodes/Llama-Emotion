"""STDP updater tests (Track C-I1 exploratory)."""

import torch
import torch.nn as nn

from src.brain.stdp import STDPUpdater


def test_stdp_ltp_pre_before_post():
    updater = STDPUpdater(a_ltp=0.1, a_ltd=0.0, w_min=-1.0, w_max=1.0)
    layer = nn.Linear(2, 2, bias=False)
    layer.weight.data.zero_()
    pre = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
    post = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    delta = updater.update_linear(layer, pre, post)
    assert delta[1, 0] > 0
    assert layer.weight[1, 0] > 0


def test_stdp_clamps_weights():
    updater = STDPUpdater(a_ltp=10.0, a_ltd=0.0, w_min=0.0, w_max=0.5)
    layer = nn.Linear(1, 1, bias=False)
    layer.weight.data.zero_()
    pre = torch.tensor([[1.0], [1.0]])
    post = torch.tensor([[1.0], [1.0]])
    updater.update_linear(layer, pre, post)
    assert layer.weight.max() <= 0.5
