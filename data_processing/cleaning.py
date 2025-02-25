import re

import pandas as pd


def clean_sentences(sentence: str) -> str:
    sentence = clean_html(sentence)
    sentence = clean_https(sentence)
    sentence = clean_www(sentence)
    sentence = clean_email(sentence)
    sentence = clean_numbers(sentence)
    sentence = clean_emoji(sentence)
    sentence = clean_twitter_at(sentence)
    sentence = clean_symbols(sentence)
    sentence = clean_double_whitespace(sentence)
    return sentence


def clean_html(sentence: str) -> str:
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, "", sentence)
    return cleantext


def clean_https(sentence: str) -> str:
    cleantext = re.sub(r"http\S+", "", sentence)
    return cleantext


def clean_www(sentence: str) -> str:
    cleantext = re.sub(r"www\S+", "", sentence)
    return cleantext


def clean_email(sentence: str) -> str:
    cleantext = re.sub(r"\S+@\S+", "", sentence)
    return cleantext


def clean_numbers(sentence: str) -> str:
    cleantext = re.sub(r"\d+", "", sentence)
    return cleantext


def clean_emoji(sentence: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    cleantext = re.sub(emoji_pattern, "", sentence)
    return cleantext


def clean_twitter_at(sentence: str) -> str:
    cleantext = re.sub(r"@\w+", "", sentence)
    return cleantext


def clean_symbols(sentence: str) -> str:
    cleantext = re.sub(r"[^\w\s]", " ", sentence)
    return cleantext


def clean_double_whitespace(sentence: str) -> str:
    cleantext = re.sub(r"\s+", " ", sentence).strip()
    return cleantext


if __name__ == "__main__":
    test = "एला आप मन अपन डिस्ट्रीब्यूसन डिस्क मं पा सकथो या एला इहां से डाउनलोड कर सकथो - http: // www. vcdimager. org"
    test2 = "+210-250% Enhanced Damage  50% Bonus To Attack Rating  +150% Damage To Demons  +150% Damage To Undead  9% Life Stolen Per Hit  (0.75/clvl) +0-75% Deadly Strike  Repairs 1 Durability In 4 Sec20% Bonus To Attack Rating  Replenish Life +15  Knockback  +50% Enhanced Damage  Adds 4-20 Cold Damage 3 Sec Duration  Adds 4-20 Cold  Damage 3 Sec Duration"
    test3 = "Public,نفس سيناريو  كوستا  يوم العاشرة 😂😂😂😂 منظنش  أنه مكلخ  يدخل  أساسي  🤔 https://t.co/Bi04uUbzTM,ar"
    print(clean_sentences(test3))
