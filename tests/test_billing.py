from __future__ import annotations

import pytest

from slotgw.billing import bill_request, per_token_usd, shape_seconds
from slotgw.config import Pricing, ProviderPricing, Shape, ShapePricing, TokenPricing

PRICING = TokenPricing(input_usd_per_million=0.15, output_usd_per_million=0.60, batch_discount=0.5)
FULL = Pricing(token=PRICING, provider=ProviderPricing(gpu_usd_per_hour=0.3), shape=ShapePricing(prompt_usd_per_million=0.0))
WITH_PREFILL = Pricing(
    token=PRICING, provider=ProviderPricing(gpu_usd_per_hour=0.3), shape=ShapePricing(prompt_usd_per_million=0.1)
)
SHAPE = Shape(name="small.8k.30tps", model="small", max_context=8192, tps=30, price_per_second=0.000018)
LAZY = Shape(
    name="small.8k.15tps.lazy", model="small", max_context=8192, tps=15, price_per_second=0.0000045, lazy=True
)


def test_per_token_reference_and_batch_discount():
    assert per_token_usd(1_000_000, 0, False, PRICING) == pytest.approx(0.15)
    assert per_token_usd(0, 1_000_000, False, PRICING) == pytest.approx(0.60)
    assert per_token_usd(1_000_000, 1_000_000, True, PRICING) == pytest.approx(0.375)


def test_shape_seconds_caps_slow_engine_and_charges_fast_delivery_as_is():
    # 300 tokens at 30 tps promised = 10 s after a 0.5 s TTFT -> bound 10.5 s
    wall, bound, billed = shape_seconds(100.0, 100.5, 112.0, 300, 30)  # engine slow: 12 s wall
    assert (wall, bound, billed) == (pytest.approx(12.0), pytest.approx(10.5), pytest.approx(10.5))
    wall, bound, billed = shape_seconds(100.0, 100.5, 106.0, 300, 30)  # burst: 6 s wall
    assert (wall, bound, billed) == (pytest.approx(6.0), pytest.approx(10.5), pytest.approx(6.0))


def test_shape_seconds_zero_without_delivery():
    assert shape_seconds(100.0, None, None, 0, 30) == (0.0, 0.0, 0.0)
    assert shape_seconds(None, 100.5, 101.0, 10, 30) == (0.0, 0.0, 0.0)


def test_bill_request_excludes_wasted_tokens_and_prices_both_ways():
    b = bill_request(
        shape=SHAPE,
        pricing=FULL,
        prompt_tokens=1000,
        output_tokens=120,
        delivered_tokens=100,
        t_admitted=0.0,
        t_first_token=0.2,
        t_last_token=3.0,
    )
    assert b.delivered_tokens == 100
    assert b.bound_s == pytest.approx(0.2 + 100 / 30)
    assert b.billed_s == pytest.approx(3.0)
    assert b.shape_usd == pytest.approx(3.0 * 0.000018)
    assert b.burst_discount_usd == pytest.approx((b.bound_s - 3.0) * 0.000018)
    assert b.per_token_usd == pytest.approx((1000 * 0.15 + 120 * 0.60) / 1e6)
    d = b.as_dict()
    assert set(d) == {
        "delivered_tokens", "wall_s", "bound_s", "billed_s", "prompt_usd", "shape_usd", "bound_usd", "wall_usd",
        "per_token_usd",
    }
    assert b.prompt_usd == 0.0


def test_lazy_shape_uses_batch_discount_for_reference():
    b = bill_request(
        shape=LAZY,
        pricing=FULL,
        prompt_tokens=1000,
        output_tokens=100,
        delivered_tokens=100,
        t_admitted=0.0,
        t_first_token=0.5,
        t_last_token=7.0,
    )
    assert b.per_token_usd == pytest.approx(0.5 * (1000 * 0.15 + 100 * 0.60) / 1e6)
    assert b.bound_s == pytest.approx(0.5 + 100 / 15)


def test_prefill_charge_adds_to_shape_bills_only():
    b = bill_request(
        shape=SHAPE, pricing=WITH_PREFILL, prompt_tokens=20_000, output_tokens=100, delivered_tokens=100,
        t_admitted=0.0, t_first_token=1.0, t_last_token=4.0,
    )
    assert b.prompt_usd == pytest.approx(0.002)
    assert b.shape_usd == pytest.approx(4.0 * 0.000018 + 0.002)  # wall 4 s < bound 4.33 s
    assert b.bound_usd == pytest.approx((1.0 + 100 / 30) * 0.000018 + 0.002)
    assert b.per_token_usd == pytest.approx((20_000 * 0.15 + 100 * 0.60) / 1e6)
    none = bill_request(
        shape=SHAPE, pricing=WITH_PREFILL, prompt_tokens=20_000, output_tokens=0, delivered_tokens=0,
        t_admitted=0.0, t_first_token=None, t_last_token=None,
    )
    assert none.shape_usd == 0.0  # nothing delivered, nothing billed
