"""Pricing one request two ways from the same meter row (docs/DESIGN.md section 8).

Per-token (reference): prompt and output tokens at list prices; lazy requests get the batch
discount. Shape (wall-clock): seconds from admission to the last token delivered, queue time
excluded, capped at the shape's promise `ttft + delivered_tokens / tps`, times the shape's
price per second. The cap is what makes a shape bill predictable: the customer never pays for the
provider's slowness, and a faster-than-promised delivery (burst) shortens the bill.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Pricing, Shape, TokenPricing


def per_token_usd(prompt_tokens: int, output_tokens: int, lazy: bool, pricing: TokenPricing) -> float:
    usd = (prompt_tokens * pricing.input_usd_per_million + output_tokens * pricing.output_usd_per_million) / 1e6
    return usd * (pricing.batch_discount if lazy else 1.0)


def shape_seconds(
    t_admitted: float | None,
    t_first_token: float | None,
    t_last_token: float | None,
    delivered_tokens: int,
    tps: float,
) -> tuple[float, float, float]:
    """(wall_s, bound_s, billed_s). A request that never delivered a token bills nothing."""
    if t_admitted is None or t_first_token is None or t_last_token is None or delivered_tokens <= 0:
        return 0.0, 0.0, 0.0
    wall_s = max(0.0, t_last_token - t_admitted)
    bound_s = max(0.0, t_first_token - t_admitted) + delivered_tokens / tps
    return wall_s, bound_s, min(wall_s, bound_s)


@dataclass(frozen=True)
class Bill:
    delivered_tokens: int
    wall_s: float
    bound_s: float
    billed_s: float
    prompt_usd: float  # prefill charge (pricing.shape.prompt_usd_per_million), included in the shape bills
    shape_usd: float
    bound_usd: float
    wall_usd: float
    per_token_usd: float

    @property
    def burst_discount_usd(self) -> float:
        """What the cap took off the bill (bound minus billed)."""
        return self.bound_usd - self.shape_usd

    def as_dict(self) -> dict[str, float | int]:
        return {
            "delivered_tokens": self.delivered_tokens,
            "wall_s": round(self.wall_s, 4),
            "bound_s": round(self.bound_s, 4),
            "billed_s": round(self.billed_s, 4),
            "prompt_usd": self.prompt_usd,
            "shape_usd": self.shape_usd,
            "bound_usd": self.bound_usd,
            "wall_usd": self.wall_usd,
            "per_token_usd": self.per_token_usd,
        }


def bill_request(
    *,
    shape: Shape,
    pricing: Pricing,
    prompt_tokens: int,
    output_tokens: int,
    delivered_tokens: int,
    t_admitted: float | None,
    t_first_token: float | None,
    t_last_token: float | None,
) -> Bill:
    """`output_tokens` is what the engine generated for the billed run; `delivered_tokens` what the
    client received. Discarded (preempted) runs and undelivered tokens are never billed."""
    delivered = max(0, delivered_tokens)
    wall_s, bound_s, billed_s = shape_seconds(t_admitted, t_first_token, t_last_token, delivered, shape.tps)
    p = shape.price_per_second
    prompt_usd = prompt_tokens * pricing.shape.prompt_usd_per_million / 1e6 if delivered > 0 else 0.0
    return Bill(
        delivered_tokens=delivered,
        wall_s=wall_s,
        bound_s=bound_s,
        billed_s=billed_s,
        prompt_usd=prompt_usd,
        shape_usd=billed_s * p + prompt_usd,
        bound_usd=bound_s * p + prompt_usd,
        wall_usd=wall_s * p + prompt_usd,
        per_token_usd=per_token_usd(prompt_tokens, output_tokens, shape.lazy, pricing.token),
    )
