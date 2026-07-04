"""Gate v3 loss unit tests (no GPU)."""

import torch

from src.train.gate_loss import distress_margin_loss, neutral_noop_loss


def test_distress_margin_when_hooks_help():
    ce_on = torch.tensor(1.0)
    ce_off = torch.tensor(2.0)
    loss = distress_margin_loss(ce_on, ce_off, margin=0.1)
    assert float(loss.item()) == 0.0


def test_distress_margin_when_hooks_hurt():
    ce_on = torch.tensor(2.5)
    ce_off = torch.tensor(2.0)
    loss = distress_margin_loss(ce_on, ce_off, margin=0.1)
    assert float(loss.item()) > 0.0


def test_neutral_noop_when_hooks_match():
    ce_on = torch.tensor(1.0)
    ce_off = torch.tensor(1.02)
    loss = neutral_noop_loss(ce_on, ce_off, eps=0.05)
    assert float(loss.item()) == 0.0


def test_neutral_noop_when_hooks_worse():
    ce_on = torch.tensor(1.5)
    ce_off = torch.tensor(1.0)
    loss = neutral_noop_loss(ce_on, ce_off, eps=0.05)
    assert float(loss.item()) > 0.0
