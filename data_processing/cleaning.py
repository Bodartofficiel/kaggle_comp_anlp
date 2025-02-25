import re

import pandas as pd


def clean_sentences(sentence: str) -> str:
    sentence = clean_html(sentence)
    sentence = clean_https(sentence)
    return sentence


def clean_html(sentence: str) -> str:
    cleanr = re.compile("<.*?>")
    cleantext = re.sub(cleanr, "", sentence)
    return cleantext


def clean_https(sentence: str) -> str:
    cleantext = re.sub(r"http\S+", "", sentence)
    return cleantext


if __name__ == "__main__":
    test_sentence = "This is a phrase with an html tag <a href='http://example.com'>link</a> and a URL http://example.com"
    cleaned_sentence = clean_sentences(test_sentence)
    print(cleaned_sentence)
