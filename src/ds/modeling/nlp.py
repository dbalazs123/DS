"""Text and LLM helpers.

Heavier NLP dependencies (currently ``tiktoken``) ship in the ``nlp`` extra:
``uv sync --extra nlp``. Functions here degrade gracefully to dependency-free
fallbacks so importing the module never fails.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tiktoken import Encoding


@cache
def _resolve_encoding(model: str) -> Encoding | None:
    """Resolve ``model`` to a tiktoken encoding, or ``None`` for the fallback.

    The outcome is cached per ``model`` for the life of the process —
    **including failure**. Loading an encoding can cost a network round-trip
    (tiktoken downloads its vocabulary on first use), and when that endpoint
    is unreachable the attempt fails slowly; re-probing on every call turned
    the first real consumer's 5,000-message feature map into a half-hour
    stall. Caching the failure makes the degradation a one-time cost and
    keeps a process on one counting path for its whole run.
    """
    try:
        import tiktoken
    except ImportError:
        return None

    try:
        try:
            return tiktoken.get_encoding(model)
        except ValueError:
            return tiktoken.encoding_for_model(model)
    except Exception:
        # tiktoken is installed but the encoding could not be loaded (e.g. the
        # vocabulary download failed with no network). Record the failure so
        # the fallback is only slow once.
        return None


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Estimate the token count of a string.

    Uses ``tiktoken`` when available for an accurate count; otherwise falls
    back to a whitespace word count so the function is always callable.

    Which path is live is resolved **once per process** for each ``model``
    and the outcome cached, success or failure. With tiktoken installed but
    its vocabulary endpoint unreachable, only the first call pays the failed
    download attempt — every later call falls back immediately — and a
    process never switches counting paths mid-run, so mapping a column of
    texts yields mutually comparable counts even if connectivity changes.

    Args:
        text: The input text.
        model: A ``tiktoken`` encoding or model name.

    Returns:
        The (estimated) number of tokens.
    """
    encoding = _resolve_encoding(model)
    if encoding is None:
        return len(text.split())
    try:
        return len(encoding.encode(text))
    except Exception:
        # A loaded encoding can still reject specific inputs (tiktoken raises
        # on special tokens such as "<|endoftext|>" in the text). Fall back to
        # the whitespace estimate for that call rather than raising — this
        # module is documented to stay callable regardless of input.
        return len(text.split())


__all__ = ["count_tokens"]
