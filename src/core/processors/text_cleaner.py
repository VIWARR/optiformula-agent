import re
import html
from typing import Final
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

RE_IMAGES: Final = re.compile(r'!\[.*?\]\(.*?\)')
RE_PAGE_NUMS: Final = re.compile(r'^\s*\d{1,3}\s*$', re.MULTILINE)
RE_SUPERSCRIPTS: Final = re.compile(r'<sup>.*?</sup>')
RE_TRAILING_SPACES: Final = re.compile(r'[ \t]+$', re.MULTILINE)
RE_MULTIPLE_NEWLINES: Final = re.compile(r'\n{3,}')
RE_SPAN_IN_HEADER: Final = re.compile(r'(#+)\s*<span[^>]*>(.*?)</span>')
RE_HTML_TAGS: Final = re.compile(r'</?(u|strong|b|span|em|i)>', re.IGNORECASE)
RE_TRAINING_LINKS: Final = re.compile(r'Тренинг \d+ Optimacros.*?https?://\S+')
RE_EXCESSIVE_SPACES: Final = re.compile(r'[ \t]+')
RE_EMPTY_LINES: Final = re.compile(r'\n\s*\n')
RE_ANY_HEADING: Final = re.compile(r'^(#{1,3}) (.+)$', re.MULTILINE)
RE_TABLE_CELL_ITALIC: Final = re.compile(r'(?<=\|)\s*\*{1,3}(.+?)\*{1,3}\s*(?=\|)')

ARTIFACTS_PATTERNS: Final = [
    re.compile(r'Optimacros logo', re.IGNORECASE),
    re.compile(r'page_\d+', re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Heading normalization
# ---------------------------------------------------------------------------

_SUBSECTION_TITLES: Final = frozenset({
    'Синтаксис', 'Аргументы', 'Возвращаемое значение',
    'Эквивалент в Excel', 'Примеры', 'Пример', 'Примечания',
    'Ограничения', 'Значащие символы', 'Примеры синтаксиса',
    'Пример 1', 'Пример 2', 'Пример 3',
})

_RE_CODE_EXAMPLE_HEADING: Final = re.compile(
    r'^(?:'
    r':\s+'
    r'|[A-ZА-ЯЁ_]{2,}\s*[\(\[]'
    r'|!'
    r'|\*Пример'
    r')',
)

_RE_FUNC_HEADING: Final = re.compile(
    r'^(?:<mark>\s*)?Функция\s+\w',
    re.IGNORECASE,
)

def normalize_headings(text: str) -> str:
    lines = text.split('\n')
    result = []

    for line in lines:
        m = RE_ANY_HEADING.match(line)
        if not m:
            result.append(line)
            continue

        hashes, title = m.group(1), m.group(2).strip()
        clean_title = re.sub(r'</?mark>', '', title).strip()

        if any(clean_title == s or clean_title.startswith(s + ' ') for s in _SUBSECTION_TITLES):
            result.append(f'### {title}')
            continue

        if _RE_FUNC_HEADING.match(clean_title):
            result.append(f'## {title}')
            continue

        if _RE_CODE_EXAMPLE_HEADING.match(clean_title):
            result.append(title)
            continue

        result.append(line)

    return '\n'.join(result)

# ---------------------------------------------------------------------------
# Main cleaner
# ---------------------------------------------------------------------------

def clean_markdown(text: str) -> str:
    if not text:
        return ""

    # 1. 
    text = html.unescape(text)
    text = RE_IMAGES.sub('', text)
    text = RE_PAGE_NUMS.sub('', text)
    text = RE_TRAINING_LINKS.sub('', text)
    for pattern in ARTIFACTS_PATTERNS:
        text = pattern.sub('', text)

    # 2. 
    soup = BeautifulSoup(text, 'html.parser')
    for table in soup.find_all('table'):
        markdown_table = []
        for tr in table.find_all('tr'):
            cells = [cell.get_text(strip=True).replace('\n', ' ') for cell in tr.find_all(['td', 'th'])]
            if any(cells):
                markdown_table.append(f"| {' | '.join(cells)} |")
        
        if markdown_table:
            table.replace_with("\n\n" + "\n".join(markdown_table) + "\n\n")
        else:
            table.decompose()

    text = soup.get_text()

    # 3.
    text = normalize_headings(text)

    # 4. 
    text = RE_SPAN_IN_HEADER.sub(r'\1 \2', text)
    text = text.replace('<br/>', ' ').replace('<br>', ' ')
    text = RE_HTML_TAGS.sub('', text)
    text = RE_SUPERSCRIPTS.sub('', text)
    text = RE_TABLE_CELL_ITALIC.sub(r'\1', text)

    # 5. 
    text = RE_TRAILING_SPACES.sub('', text)
    text = RE_EXCESSIVE_SPACES.sub(' ', text)
    text = RE_EMPTY_LINES.sub('\n\n', text)
    text = RE_MULTIPLE_NEWLINES.sub('\n\n', text)

    return text.strip()