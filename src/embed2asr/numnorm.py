#!/usr/bin/env python3
"""Deterministic number verbalization (digits -> spoken words) applied to RAW text, before common.norm.
zh: years 19xx/20xx -> digit-wise (2023年 -> 二零二三年), counts -> cardinal (10月 -> 十月, 2個 -> 兩個), N% -> 百分之N,
    decimals (5.8億 -> 五點八億), thousands separators (5 , 000 -> 五千), leading-0 strings (0988 -> 零九八八).
en: num2words 0.5.14 (years via to='year', ordinals, decimals, %, $/£/€/¥ amounts, 1990s).
Uncertain shapes (times 9:30, ranges 3-5, fractions/dates 1/2, digits glued to latin letters G2G/150MW/COVID-19, ...) are
left UNCHANGED and counted as skipped, so downstream digit filters still drop those rows.
API: zh(text)->(text, n_skipped); en(text)->(text, n_skipped); norm_nn(text, lang) = common.norm(numnorm(text)):
use norm_nn on BOTH reference and hypothesis (e.g. Whisper output) when scoring against *_nn references.
"""

import re

from .normalize import norm

D = "零一二三四五六七八九"
UNITS = ("", "萬", "億", "兆")
# measure words after which a bare 2 is read 兩 (2個 -> 兩個); 月/日/號/樓 and 第N keep 二
LIANG = set("個位人天次件家種倍名年歲項條所座台臺部架艘場間本張隻棟塊元週周小分秒成千百萬億兆點顆輛棵層")
COUNT_AFTER = LIANG | set("多餘萬億兆千百%")  # a 19xx/20xx number followed by these is a quantity, not a year
FW = str.maketrans("０１２３４５６７８９％．，", "0123456789%.,")


def _sec(n):  # 1..9999
    out, zero = "", False
    for p, u in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        d = n // p % 10
        if d == 0:
            zero = bool(out)
            continue
        out += ("零" if zero else "") + D[d] + u
        zero = False
    return out


def zh_cardinal(n: int) -> str:
    if n == 0:
        return "零"
    gs = []
    while n:
        gs.append(n % 10000)
        n //= 10000
    out, gap = "", False
    for i in reversed(range(len(gs))):
        g = gs[i]
        if g == 0:
            gap = bool(out)
            continue
        if out and (gap or g < 1000):
            out += "零"
        out += _sec(g) + UNITS[i]
        gap = False
    out = re.sub(r"(?<!十)二(?=[百千萬億兆])", "兩", out)
    return out[1:] if out.startswith("一十") else out


def zh_digits(s: str) -> str:
    return "".join(D[int(c)] for c in s)


def zh_number(s: str) -> str:  # "5000" / "5.80"
    i, _, f = s.partition(".")
    return zh_cardinal(int(i)) + ("點" + zh_digits(f) if f else "")


ZH_NUM = re.compile(r"\d+(?:\s?,\s?\d{3})+(?![\d.])|\d+(?:\.\d+)?")


def zh(text: str):
    text = text.translate(FW)
    skipped = 0

    def sub(m):
        nonlocal skipped
        s, a, b = m.group(0), m.string[: m.start()], m.string[m.end() :]
        prev, nxt = a.rstrip()[-1:], b.lstrip()[:1]
        tight_prev, tight_next = a[-1:], b[:1]
        num = re.sub(r"\s|,", "", s)
        if (
            re.match(r"[A-Za-z]", tight_prev)
            or re.match(r"[A-Za-z]", tight_next)  # G2G, 150MW, PM2.5
            or (prev and prev in ":：/-~～－")
            or (nxt and nxt in ":：/-~～－")
            or len(num.split(".")[0]) > 12
        ):  # 9:30, 1/2, 3-5, ids
            skipped += 1
            return s
        if nxt == "%":
            return "百分之" + zh_number(num) + " "
        if "." in num:
            return zh_number(num)
        if num[0] == "0" and len(num) > 1:  # phone / codes: digit-wise
            return zh_digits(num)
        if len(num) == 4 and nxt == "年":
            return zh_digits(num)
        if len(num) == 4 and 1900 <= int(num) <= 2099 and "," not in s:
            if nxt in COUNT_AFTER:
                return zh_cardinal(int(num))
            return zh_digits(num)  # bare 2030 / 2022 、 2023 -> year reading
        if num == "2" and nxt in LIANG and prev != "第":
            return "兩"
        return zh_cardinal(int(num))

    out = ZH_NUM.sub(sub, text)
    return re.sub(r"(百分之[^%\s]*)\s*%", r"\1", out), skipped


# ---------------- English ----------------
MONTHS = {
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}
CUR = {"US$": "dollars", "$": "dollars", "£": "pounds", "€": "euros", "¥": "yen"}
SCALE = r"(?:\s(?P<scale>thousand|million|billion|trillion))?"
EN_NUM = re.compile(r"(?P<cur>US\$|\$|£|€|¥)?(?P<num>\d+(?:,\d{3})*(?:\.\d+)?)(?P<suf>st|nd|rd|th|s|%)?(?(cur)" + SCALE + ")")


def _w(x, **kw):
    from num2words import num2words

    return num2words(x, lang="en", **kw).replace("-", " ").replace(",", "")


def en_number(s: str) -> str:
    i, _, f = s.partition(".")
    return _w(int(i)) + (" point " + " ".join(_w(int(c)) for c in f) if f else "")


def en(text: str):
    skipped = 0

    def sub(m):
        nonlocal skipped
        s, a, b = m.group(0), m.string[: m.start()], m.string[m.end() :]
        num, suf, cur = m.group("num").replace(",", ""), m.group("suf"), m.group("cur")
        if (
            (re.match(r"[A-Za-z0-9]", a[-1:]) and not cur)
            or re.match(r"[A-Za-z0-9]", b[:1])
            or (a[-1:] and a[-1:] in ":/")
            or (b[:1] and b[:1] in ":/")
            or re.match(r"[-–—]\d", b[:2])
            or re.search(r"[A-Za-z0-9][-–—]$", a[-2:])
        ):
            skipped += 1
            return s  # 9am, km2, COVID-19, 9:30, 1/2, ranges 100-200
        if cur:
            if suf:
                skipped += 1
                return s
            sc = m.group("scale")
            unit = CUR[cur] if not (num == "1" and not sc) else CUR[cur].rstrip("s")
            return f"{en_number(num)}{' ' + sc if sc else ''} {unit}"
        if suf in ("st", "nd", "rd", "th"):
            return _w(int(num), to="ordinal") if "." not in num else s
        if suf == "%":
            return en_number(num) + " percent"
        if suf == "s":
            if len(num) == 4 and "." not in num:
                y = _w(int(num), to="year")
                return y[:-1] + "ies" if y.endswith("y") else y + "s"
            skipped += 1
            return s
        if "." in num:
            return en_number(num)
        near = re.findall(r"[A-Za-z]+", a[-12:])[-1:] + re.findall(r"[A-Za-z]+", b[:12])[:1]
        if int(num) <= 31 and any(w.lower() in MONTHS for w in near):  # 24 August / March 20 -> ordinal
            return _w(int(num), to="ordinal")
        if len(num) == 4 and "," not in m.group("num") and 1100 <= int(num) <= 2099:
            return _w(int(num), to="year")
        return _w(int(num))

    return EN_NUM.sub(sub, text), skipped


def numnorm(text: str, lang: str):
    return (en if lang.startswith("en") else zh)(text)


def norm_nn(text: str, lang: str = "zh") -> str:
    """Scoring helper: apply to reference AND hypothesis alike."""
    return norm(numnorm(text, lang)[0])
