import random
import re
import unicodedata
from typing import List

# Seed slang dictionary (extend this with your observed tokens)
SLANG_DICT = {
    'ko': 'không',
    'k': 'không',
    'hok': 'không',
    'hokk': 'không',
    'j': 'gì',
    'nj': 'ngon',
    'hj': 'hihi',
    'z': 'z',
    'xau': 'xấu',
    'depjjj': 'đẹp'
}


def remove_diacritics(text: str) -> str:
    nfkd = unicodedata.normalize('NFD', text)
    return ''.join([c for c in nfkd if not unicodedata.combining(c)])


def collapse_repeats(text: str) -> str:
    # collapse repeated characters longer than 2 to 2
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def random_delete_chars(text: str, p: float = 0.03) -> str:
    if p <= 0:
        return text
    out = []
    for ch in text:
        if random.random() < p and not ch.isspace():
            continue
        out.append(ch)
    return ''.join(out)


def remove_random_spaces(text: str, p: float = 0.05) -> str:
    # randomly remove spaces between words with probability p
    parts = text.split(' ')
    if len(parts) <= 1:
        return text
    out = [parts[0]]
    for w in parts[1:]:
        if random.random() < p:
            out[-1] = out[-1] + w
        else:
            out.append(w)
    return ' '.join(out)


def apply_slang_substitution(text: str, slang_map: dict = None) -> str:
    slang_map = slang_map or SLANG_DICT
    # word-boundary replacement (case-insensitive)
    def repl(m):
        w = m.group(0)
        return slang_map.get(w.lower(), w)

    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in slang_map.keys()) + r")\b", flags=re.IGNORECASE)
    return pattern.sub(repl, text)


def add_random_teencode(text: str, p_drop_diacritic=0.3, p_delete_char=0.02, p_remove_space=0.03,
                         p_repeat_chars=0.02, apply_slang=True) -> str:
    s = text
    # 1. randomly drop diacritics
    if random.random() < p_drop_diacritic:
        s = remove_diacritics(s)

    # 2. random character deletions
    s = random_delete_chars(s, p_delete_char)

    # 3. remove some spaces
    if random.random() < p_remove_space:
        s = remove_random_spaces(s, p_remove_space)

    # 4. collapse repeated characters
    if random.random() < p_repeat_chars:
        s = collapse_repeats(s)

    # 5. slang substitution (to simulate informal tokens)
    if apply_slang:
        s = apply_slang_substitution(s)

    return s


def generate_noisy_variants(text: str, n: int = 1) -> List[str]:
    variants = []
    for _ in range(n):
        # randomize parameters a bit per variant
        p_drop = random.uniform(0.1, 0.6)
        p_del = random.uniform(0.0, 0.06)
        p_space = random.uniform(0.0, 0.08)
        p_rep = random.uniform(0.0, 0.2)
        v = add_random_teencode(text, p_drop_diacritic=p_drop, p_delete_char=p_del,
                                 p_remove_space=p_space, p_repeat_chars=p_rep)
        variants.append(v)
    return variants


if __name__ == '__main__':
    # quick local test
    s = 'Điện thoại đẹp lắm, pin trâu và chụp ảnh rất nét!'
    for v in generate_noisy_variants(s, 5):
        print(v)
