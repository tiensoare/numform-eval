"""
Parse a model's free-text answer into a number and score it against ground truth.

Handles the common shapes models emit:
  - "ANSWER: 1.0472"          (the format we request)
  - fractions: "22/7"
  - symbolic: "pi/3", "sqrt(2)"  (parsed via sympy)
  - a bare number on the last line, if the ANSWER tag is missing
"""

import re
from typing import Optional

import sympy as sp


ANSWER_RE = re.compile(r"ANSWER\s*:\s*(.+)", re.IGNORECASE)


def extract_answer_text(response: str) -> Optional[str]:
    """Pull the answer string out of the model response."""
    if not response:
        return None
    # prefer the explicit ANSWER: tag (last occurrence wins)
    matches = ANSWER_RE.findall(response)
    if matches:
        return matches[-1].strip()
    # fallback: last non-empty line
    lines = [ln.strip() for ln in response.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def to_float(answer_text: str) -> Optional[float]:
    """
    Convert an answer string to a float. Tries plain float, then fraction,
    then sympy symbolic parsing (so 'pi/3' or 'sqrt(2)' evaluate correctly).
    """
    if answer_text is None:
        return None
    s = answer_text.strip().rstrip(".")
    # strip surrounding text like "x = 1.047" or "approximately 1.047"
    s = s.replace("=", " ").split()[-1] if "=" in s else s

    # plain float
    try:
        return float(s)
    except ValueError:
        pass

    # sympy handles fractions AND symbols: "22/7", "pi/3", "sqrt(2)"
    try:
        expr = sp.sympify(s, locals={"pi": sp.pi, "sqrt": sp.sqrt, "e": sp.E, "E": sp.E})
        return float(expr)
    except Exception:
        return None


def score(response: str, truth_float: float, tol: float) -> dict:
    """
    Returns dict: parsed value, correctness, and absolute/relative error.
    'correct' uses relative tolerance when truth is not near zero.
    """
    ans_text = extract_answer_text(response)
    val = to_float(ans_text) if ans_text else None

    if val is None:
        return {
            "answer_text": ans_text,
            "parsed": None,
            "correct": False,
            "abs_err": None,
            "rel_err": None,
            "parse_ok": False,
        }

    abs_err = abs(val - truth_float)
    denom = max(abs(truth_float), 1e-12)
    rel_err = abs_err / denom
    correct = (abs_err <= tol) or (rel_err <= tol)
    return {
        "answer_text": ans_text,
        "parsed": val,
        "correct": bool(correct),
        "abs_err": abs_err,
        "rel_err": rel_err,
        "parse_ok": True,
    }


if __name__ == "__main__":
    truth = float(sp.pi / 3)  # 1.0472
    for r in [
        "work work\nANSWER: 1.0472",
        "the result is pi/3\nANSWER: pi/3",
        "ANSWER: 22/7",          # wrong
        "1.0472",                # bare
        "I cannot solve this",   # unparseable
    ]:
        print(repr(r[:30]), "->", score(r, truth, 1e-3))
