"""Shared helpers for calling Claude: one client, model names, and cost per call."""

from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # puts ANTHROPIC_API_KEY into the environment

OPUS = "claude-opus-5-5"
HAIKU = "claude-haiku-5-5"

# US dollars per 1 million tokens: (input, output)
PRICES = {
    OPUS: (4.00, 20.00),
    HAIKU: (0.10, 0.50),
}

_client = None


def client():
    """Create the API client once and reuse it."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def cost(model, usage):
    """Dollar cost of one API call, from the token counts the API returns."""
    price_in, price_out = PRICES[model]
    return (usage.input_tokens * price_in + usage.output_tokens * price_out) / 1_000_000
