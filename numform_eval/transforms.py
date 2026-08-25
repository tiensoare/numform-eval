"""
Number representation transforms.

Four ways to present a numeric value to a model:
  - original   : the plain decimal / float, as-is (control)
  - exact      : Fraction(str(val))                 -> lossless but huge p/q
  - limit      : limit_denominator(N)               -> clean bounded fraction
  - nsimplify  : symbolic recovery (pi, sqrt(2)...) -> falls back to a fraction

Design rule learned the hard way: ALWAYS transform from the highest-precision
value available. Rationalizing an already-truncated float bakes in its error.
So each Number can carry an optional `exact_expr` (a sympy expression) that is
the true source of truth; the float is only a display/fallback.
"""

from dataclasses import dataclass
from fractions import Fraction
from typing import Optional

import sympy as sp
from sympy import nsimplify, N, pi, sqrt, E

# Constants nsimplify is allowed to recognise. Extend as needed.
DEFAULT_CONSTS = [pi, sqrt(2), sqrt(3), sqrt(5), E]


@dataclass
class Number:
    """A single numeric value plus its provenance."""
    label: str                      # human label, e.g. "pi" or "1/3"
    value: float                    # float form (display / fallback)
    exact_expr: Optional[sp.Expr] = None  # true high-precision source, if known

    def high_precision(self, digits: int = 30) -> str:
        """Best available decimal string. Uses exact_expr when present."""
        if self.exact_expr is not None:
            return str(N(self.exact_expr, digits))
        return repr(self.value)


def as_original(num: Number) -> str:
    """The plain form: just the number as a decimal string."""
    if num.exact_expr is not None:
        # show a normal-looking decimal, not 30 digits
        return str(N(num.exact_expr, 15))
    return repr(num.value)


def as_exact(num: Number) -> str:
    """Exact decimal -> p/q. Lossless, large denominators."""
    frac = Fraction(str(num.value))
    return f"{frac.numerator}/{frac.denominator}"


def as_limit(num: Number, max_den: int = 1000) -> str:
    """Best fraction with denominator <= max_den (continued-fraction convergent)."""
    src = num.high_precision(30)
    frac = Fraction(src).limit_denominator(max_den)
    return f"{frac.numerator}/{frac.denominator}"


def as_nsimplify(num: Number, consts=DEFAULT_CONSTS, tolerance: float = 1e-10) -> str:
    """
    Symbolic recovery. Returns a symbol (pi, sqrt(2), pi/4) when one fits;
    otherwise falls back to a bounded fraction. Guards against fabricated
    closed-forms (too many terms / symbols).
    """
    src = num.high_precision(30)
    try:
        sym = nsimplify(src, consts, tolerance=tolerance)
    except Exception:
        return as_limit(num)

    # reject fabricated / overly-complex closed forms
    if len(sym.free_symbols) > 1 or sym.count_ops() > 3:
        return as_limit(num)
    return str(sym)


# registry so the runner can iterate over methods by name
METHODS = {
    "original":  as_original,
    "exact":     as_exact,
    "limit":     as_limit,
    "nsimplify": as_nsimplify,
}


def render(num: Number, method: str, **kwargs) -> str:
    """Render one Number under one method name."""
    if method not in METHODS:
        raise KeyError(f"unknown method {method!r}; choices: {list(METHODS)}")
    return METHODS[method](num, **kwargs) if kwargs else METHODS[method](num)


if __name__ == "__main__":
    import math
    nums = [
        Number("1/3", 1/3, sp.Rational(1, 3)),
        Number("pi", math.pi, pi),
        Number("sqrt(2)", math.sqrt(2), sqrt(2)),
        Number("pi/4", math.pi/4, pi/4),
        Number("e", math.e, E),
        Number("22/7", 22/7, sp.Rational(22, 7)),
        Number("messy", 3.2145678923447923),  # no exact_expr on purpose
    ]
    hdr = ("label", "original", "exact", "limit", "nsimplify")
    print("{:<10}{:<18}{:<28}{:<14}{}".format(*hdr))
    print("-" * 82)
    for n in nums:
        print("{:<10}{:<18}{:<28}{:<14}{}".format(
            n.label, as_original(n),
            (as_exact(n)[:25] + "...") if len(as_exact(n)) > 25 else as_exact(n),
            as_limit(n), as_nsimplify(n)))
