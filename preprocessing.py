"""
Text preprocessing utilities for the Fake News Detection System.

Provides reusable functions for cleaning and normalizing news article text
before TF-IDF vectorization and model inference.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure required NLTK resources are available.
for resource in ("stopwords", "punkt", "punkt_tab"):
    try:
        nltk.data.find(f"corpora/{resource}" if resource == "stopwords" else f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

STEMMER = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    return re.sub(r"<[^>]+>", " ", text)


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(
        r"http\S+|www\S+|https\S+",
        " ",
        text,
        flags=re.MULTILINE,
    )


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters."""
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(text: str) -> str:
    """Remove numeric digits."""
    return re.sub(r"\d+", " ", text)


def remove_special_characters(text: str) -> str:
    """Keep only alphabetic characters and spaces."""
    return re.sub(r"[^a-zA-Z\s]", " ", text)


def remove_extra_whitespace(text: str) -> str:
    """Normalize whitespace to single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def preprocess_text(text: str) -> str:
    """
    Apply the full preprocessing pipeline to a single text string.

    Steps:
        1. Lowercase conversion
        2. HTML tag removal
        3. URL removal
        4. Punctuation removal
        5. Number removal
        6. Special character removal
        7. Stopword removal
        8. Porter stemming
        9. Extra whitespace cleanup

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Cleaned and stemmed text.
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    cleaned = text.lower()
    cleaned = remove_html_tags(cleaned)
    cleaned = remove_urls(cleaned)
    cleaned = remove_punctuation(cleaned)
    cleaned = remove_numbers(cleaned)
    cleaned = remove_special_characters(cleaned)

    words = cleaned.split()
    words = [
        STEMMER.stem(word)
        for word in words
        if word not in STOP_WORDS and len(word) > 1
    ]

    cleaned = " ".join(words)
    return remove_extra_whitespace(cleaned)


def combine_title_text(title: str, text: str) -> str:
    """Merge title and body text into a single string."""
    title = title if isinstance(title, str) else ""
    text = text if isinstance(text, str) else ""
    return f"{title} {text}".strip()


def get_text_stats(text: str, words_per_minute: int = 200) -> dict:
    """
    Compute basic text statistics for UI display.

    Parameters
    ----------
    text : str
        Input text.
    words_per_minute : int
        Average reading speed used for reading time estimation.

    Returns
    -------
    dict
        Dictionary with word_count, character_count, and reading_time_minutes.
    """
    if not isinstance(text, str):
        text = ""

    word_count = len(text.split())
    character_count = len(text)
    reading_time = max(word_count / words_per_minute, 0.1)

    return {
        "word_count": word_count,
        "character_count": character_count,
        "reading_time_minutes": round(reading_time, 1),
    }
