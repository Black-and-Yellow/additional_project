"""Conservative checks that complement NLI."""
import re
from decimal import Decimal


def claims(answer):
    protected = re.sub(r'\b(Dr|Mr|Mrs|Ms|Prof|St)\.', r'\1<dot>', answer)
    return [s.replace('<dot>', '.').strip() for s in re.split(r'(?<=[.!?])\s+|\n+', protected) if s.strip()]


def numbers(text):
    found = set()
    def time(match):
        found.add(f'time:{int(match[1])}:{match[2]}')
        return ' '
    remainder = re.sub(r'\b(\d{1,2}):(\d{2})\b', time, text)
    for value in re.findall(r'(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?\b', remainder):
        found.add(str(Decimal(value.replace(',', '')).normalize()))
    return found


def numeric_support(claim, context):
    return numbers(claim).issubset(numbers(context))
