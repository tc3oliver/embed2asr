"""Text normalization shared by training, scoring and decoding (identical to the research pipeline)."""

import re
import unicodedata

CJK = r"\u2e80-\u9fff\uf900-\ufaff\u3400-\u4dbf"  # explicit escapes: the compat-ideograph start U+F900 is NFKC-unstable as a literal
_cjk_gap = re.compile(rf"\s+(?=[{CJK}])|(?<=[{CJK}])\s+")


def norm(text: str) -> str:
    """NFKC, lowercase, 臺->台, drop punctuation/symbols (keep apostrophes), no spaces adjacent to CJK characters."""
    t = unicodedata.normalize("NFKC", text).lower().replace("臺", "台")
    t = "".join(" " if c.isspace() else c for c in t if unicodedata.category(c)[0] not in "PS" or c in "'")
    t = " ".join(t.split())
    return _cjk_gap.sub("", t)
