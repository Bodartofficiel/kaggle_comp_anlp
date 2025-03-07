import json
import re
from typing import List

import pandas as pd
from transformers import BertTokenizer

english_words = open("./languages/en_full.txt", "r")
english_words = set(line.split()[0] for line in english_words)

french_words = open("./languages/fr_full.txt", "r")
french_words = set(
    line.split()[0] for line in french_words if len(line.split()[0]) >= 4
)

spanish_words = open("./languages/es_full.txt", "r")
spanish_words = set(line.split()[0] for line in spanish_words)


def clean_sentences(row) -> str:

    sentence = row["Text"].lower()
    try:
        label = row["label"]
    except KeyError:
        label == None

    sentence = clean_html(sentence)
    sentence = clean_https(sentence)
    sentence = clean_www(sentence)
    sentence = clean_email(sentence)
    sentence = clean_numbers(sentence)
    sentence = clean_emoji(sentence)
    sentence = clean_twitter_at(sentence)
    sentence = clean_symbols(sentence)
    sentence = clean_double_whitespace(sentence)

    if len(sentence) <= 1:
        return ""
    return sentence


def clean_languages(sentence: str):
    tokenized = sentence.lower().split()
    original_sentence_length = tokenized
    tokenized = clean_english_from_sentence(tokenized, original_sentence_length)
    tokenized = clean_french_from_sentences(tokenized, original_sentence_length)
    tokenized = clean_spanish_from_sentences(tokenized, original_sentence_length)
    sentence = " ".join(tokenized)
    return sentence


def clean_html(sentence: str) -> str:
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, " ", sentence)
    return cleantext


def clean_https(sentence: str) -> str:
    cleantext = re.sub(r"http\S+", " ", sentence)
    return cleantext


def clean_www(sentence: str) -> str:
    cleantext = re.sub(r"www\S+", " ", sentence)
    return cleantext


def clean_email(sentence: str) -> str:
    cleantext = re.sub(r"\S+@\S+", " ", sentence)
    return cleantext


def clean_numbers(sentence: str) -> str:
    cleantext = re.sub(r"\d+", " ", sentence)
    return cleantext


def clean_emoji(sentence: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001f600-\U0001f64f"  # emoticons
        # "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001f680-\U0001f6ff"  # transport & map symbols
        "\U0001f1e0-\U0001f1ff"  # flags (iOS)
        # "\U00002702-\U000027B0"
        # "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    cleantext = re.sub(emoji_pattern, " ", sentence)
    return cleantext


def clean_twitter_at(sentence: str) -> str:
    cleantext = re.sub(r"@\w+", " ", sentence)
    return cleantext


def clean_symbols(sentence: str) -> str:
    sentence = re.sub(r"(?<!\w)'(?!\w)", " ", sentence)
    cleantext = re.sub(r"[^\w\s']", " ", sentence)
    return cleantext


def clean_double_whitespace(sentence: str) -> str:
    cleantext = re.sub(r"\s+", " ", sentence).strip()
    return cleantext


def clean_english_from_sentence(
    tokenized: List[str], original_sentence_length: int
) -> List[str]:
    new_sentence = []
    for token in tokenized:
        if token not in english_words:
            new_sentence.append(token)
    if len(new_sentence) < 0.5 * original_sentence_length:
        # This means that the sentence is probably in english
        return tokenized
    return new_sentence


def clean_french_from_sentences(
    tokenized: List[str], original_sentence_length: int
) -> List[str]:
    new_sentence = []
    for token in tokenized:
        if token not in french_words:
            new_sentence.append(token)
    if len(new_sentence) < 0.5 * original_sentence_length:
        return tokenized
    return new_sentence


def clean_spanish_from_sentences(
    tokenized: List[str], original_sentence_length: int
) -> List[str]:
    new_sentence = []
    for token in tokenized:
        if token in token in spanish_words:
            continue
        new_sentence.append(token)
    if len(new_sentence) < 0.5 * original_sentence_length:
        # This means that the sentence is probably in spanish
        return tokenized
    return new_sentence


if __name__ == "__main__":

    def conv_to_df(setence):
        data = {"Text": [setence], "label": ["eng"]}
        return pd.DataFrame(data)

    test = "एला आप मन अपन डिस्ट्रीब्यूसन डिस्क मं पा सकथो या एला इहां से डाउनलोड कर सकथो - http: // www. vcdimager. org"
    test2 = "+210-250% Enhanced Damage  50% Bonus To Attack Rating  +150% Damage To Demons  +150% Damage To Undead  9% Life Stolen Per Hit  (0.75/clvl) +0-75% Deadly Strike  Repairs 1 Durability In 4 Sec20% Bonus To Attack Rating  Replenish Life +15  Knockback  +50% Enhanced Damage  Adds 4-20 Cold Damage 3 Sec Duration  Adds 4-20 Cold  Damage 3 Sec Duration"
    test3 = "Public,نفس سيناريو  كوستا  يوم العاشرة 😂😂😂😂 منظنش  أنه مكلخ  يدخل  أساسي  🤔 https://t.co/Bi04uUbzTM,ar"
    test4 = "十二月十五，美國寘無人潜航器於漲海，而中國海軍穫之。"
    test5 = "This is a test sentence with some non-English words like bonjour and hola."
    test6 = "Alors que le soleil se couchait sobre el horizonte, painting the sky with shades of orange y reflejándose en el agua tranquila, un homme marchait lentamente por la playa, lost in his thoughts, recordando con nostalgia los momentos felices de su infancia, the laughter shared with old friends, les aventures vécues en terres inconnues, y preguntándose, while listening to the waves breaking on the shore, si el tiempo realmente cambia everything ou si, au fond, tout reste éternellement igual."

    # print(clean_sentences(test3))
    # print(clean_sentences(test4))
    print(conv_to_df(test6).apply(clean_sentences, axis=1)[0])
