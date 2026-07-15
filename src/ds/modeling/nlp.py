"""Text and LLM helpers.

Heavier NLP dependencies (``tiktoken``, ``sentence-transformers``, the Anthropic
SDK) ship in the ``nlp`` extra: ``uv sync --extra nlp``. Functions here degrade
gracefully to dependency-free fallbacks so importing the module never fails.
"""

from __future__ import annotations


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Estimate the token count of a string.

    Uses ``tiktoken`` when available for an accurate count; otherwise falls back
    to a whitespace word count so the function is always callable.

    Args:
        text: The input text.
        model: A ``tiktoken`` encoding or model name.

    Returns:
        The (estimated) number of tokens.
    """
    try:
        import tiktoken
    except ImportError:
        return len(text.split())

    try:
        encoding = tiktoken.get_encoding(model)
    except ValueError:
        encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


__all__ = ["count_tokens"]
