from __future__ import annotations

import html
import re
from typing import Final

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Pre-compiled regular expressions
# ---------------------------------------------------------------------------

RE_IMAGES: Final = re.compile(r"!\[.*?\]\(.*?\)")
RE_PAGE_NUMS: Final = re.compile(r"^\s*\d{1,3}\s*$", re.MULTILINE)
RE_SUPERSCRIPTS: Final = re.compile(r"<sup>.*?</sup>")
RE_TRAILING_SPACES: Final = re.compile(r"[ \t]+$", re.MULTILINE)
RE_MULTIPLE_NEWLINES: Final = re.compile(r"\n{3,}")
RE_SPAN_IN_HEADER: Final = re.compile(r"(#+)\s*<span[^>]*>(.*?)</span>")
RE_HTML_INLINE_TAGS: Final = re.compile(r"</?(u|strong|b|span|em|i)>", re.IGNORECASE)
RE_TRAINING_LINKS: Final = re.compile(r"Тренинг \d+ Optimacros.*?https?://\S+")
RE_EXCESSIVE_SPACES: Final = re.compile(r"[ \t]+")
RE_EMPTY_LINES: Final = re.compile(r"\n\s*\n")
RE_ANY_HEADING: Final = re.compile(r"^(#{1,3}) (.+)$", re.MULTILINE)
RE_TABLE_CELL_ITALIC: Final = re.compile(r"(?<=\|)\s*\*{1,3}(.+?)\*{1,3}\s*(?=\|)")

# Artefact patterns that should be stripped entirely from the output.
ARTEFACT_PATTERNS: Final[list[re.Pattern]] = [
    re.compile(r"Optimacros logo", re.IGNORECASE),
    re.compile(r"page_\d+", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Heading normalisation
# ---------------------------------------------------------------------------

# Headings whose titles are always rendered as H3 regardless of original level.
_SUBSECTION_TITLES: Final = frozenset(
    {
        "Синтаксис",
        "Аргументы",
        "Возвращаемое значение",
        "Эквивалент в Excel",
        "Примеры",
        "Пример",
        "Примечания",
        "Ограничения",
        "Значащие символы",
        "Примеры синтаксиса",
        "Пример 1",
        "Пример 2",
        "Пример 3",
    }
)

# Headings that look like inline code examples — strip the '#' prefix entirely.
_RE_CODE_EXAMPLE_HEADING: Final = re.compile(
    r"^(?:"
    r":\s+"
    r"|[A-ZА-ЯЁ_]{2,}\s*[\(\[]"
    r"|!"
    r"|\*Пример"
    r")",
)

# Function headings are always H2.
_RE_FUNC_HEADING: Final = re.compile(
    r"^(?:<mark>\s*)?Функция\s+\w",
    re.IGNORECASE,
)


def normalize_headings(text: str) -> str:
    """
    Normalise heading levels according to domain-specific rules:

    - Known subsection titles (e.g. ``Синтаксис``, ``Аргументы``) → H3.
    - Function headings matching ``Функция <name>`` → H2.
    - Code-example headings (all-caps, starts with ``:``, etc.) → plain text.
    - All other headings are left unchanged.
    """
    lines = text.split("\n")
    result: list[str] = []

    for line in lines:
        match = RE_ANY_HEADING.match(line)
        if not match:
            result.append(line)
            continue

        hashes, title = match.group(1), match.group(2).strip()
        clean_title = re.sub(r"</?mark>", "", title).strip()

        if any(
            clean_title == s or clean_title.startswith(s + " ")
            for s in _SUBSECTION_TITLES
        ):
            result.append(f"### {title}")
        elif _RE_FUNC_HEADING.match(clean_title):
            result.append(f"## {title}")
        elif _RE_CODE_EXAMPLE_HEADING.match(clean_title):
            result.append(title)  # demote to plain text
        else:
            result.append(line)

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Main cleaner
# ---------------------------------------------------------------------------

def clean_markdown(text: str) -> str:
    """
    Return a cleaned version of *text* suitable for downstream chunking.

    Pipeline (in order):
    1. Unescape HTML entities; strip images, bare page numbers, training links,
       and known LlamaCloud artefacts.
    2. Convert residual HTML ``<table>`` elements to Markdown pipe tables via
       BeautifulSoup.
    3. Normalise heading levels with :func:`normalize_headings`.
    4. Strip remaining inline HTML tags (``<span>``, ``<br>``, ``<sup>``, etc.).
    5. Collapse redundant whitespace and blank lines.

    Args:
        text: Raw Markdown string, possibly containing HTML fragments.

    Returns:
        Cleaned Markdown string, or an empty string if *text* is falsy.
    """
    if not text:
        return ""

    # Step 1 — remove noise
    text = html.unescape(text)
    text = RE_IMAGES.sub("", text)
    text = RE_PAGE_NUMS.sub("", text)
    text = RE_TRAINING_LINKS.sub("", text)
    for pattern in ARTEFACT_PATTERNS:
        text = pattern.sub("", text)

    # Step 2 — convert HTML tables to Markdown
    soup = BeautifulSoup(text, "html.parser")
    for table in soup.find_all("table"):
        rows: list[str] = []
        for tr in table.find_all("tr"):
            cells = [
                cell.get_text(strip=True).replace("\n", " ")
                for cell in tr.find_all(["td", "th"])
            ]
            if any(cells):
                rows.append(f"| {' | '.join(cells)} |")

        if rows:
            table.replace_with("\n\n" + "\n".join(rows) + "\n\n")
        else:
            table.decompose()

    text = soup.get_text()

    # Step 3 — normalise headings
    text = normalize_headings(text)

    # Step 4 — strip remaining inline HTML
    text = RE_SPAN_IN_HEADER.sub(r"\1 \2", text)
    text = text.replace("<br/>", " ").replace("<br>", " ")
    text = RE_HTML_INLINE_TAGS.sub("", text)
    text = RE_SUPERSCRIPTS.sub("", text)
    text = RE_TABLE_CELL_ITALIC.sub(r"\1", text)

    # Step 5 — normalise whitespace
    text = RE_TRAILING_SPACES.sub("", text)
    text = RE_EXCESSIVE_SPACES.sub(" ", text)
    text = RE_EMPTY_LINES.sub("\n\n", text)
    text = RE_MULTIPLE_NEWLINES.sub("\n\n", text)

    return text.strip()