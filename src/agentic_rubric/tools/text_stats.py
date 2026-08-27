"""Pure-Python text measurement. No LLM, no network, fully deterministic.

These numbers exist to keep the agent honest. An LLM judge asked "is this prose
clear?" will produce a confident answer either way; a Flesch score of 31 and
nine hedging phrases is a fact it has to argue against. Running them in Perceive
means every iteration starts from at least one measurement that did not come
from a model.

The formulas are standard and approximate -- syllable counting in English is a
heuristic, not a solved problem. They are used for *relative* movement between
iterations, which is what the loop needs, not for absolute grading.
"""

from __future__ import annotations

import re
import statistics

from ..core.state import TextMetrics

# Deliberately conservative lists: a false positive teaches the agent to delete
# a word it should have kept, which is worse than missing one.
HEDGES = (
    "somewhat", "arguably", "it could be said", "perhaps", "possibly",
    "generally believed", "in a sense", "seems that", "seem to", "tends to",
    "relatively", "fairly", "rather", "probably", "to some extent",
    "more or less", "in some ways", "it appears that", "one might argue",
)

FILLERS = (
    "there is", "there are", "the fact that", "in order to",
    "it is important to note", "needless to say", "basically", "actually",
    "very", "really", "quite", "just", "in terms of", "at the end of the day",
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])|\n{2,}")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_PASSIVE = re.compile(
    r"\b(?:is|are|was|were|be|been|being)\s+(?:\w+ly\s+)?\w+(?:ed|en)\b", re.IGNORECASE
)
_VOWEL_GROUP = re.compile(r"[aeiouy]+")


def sentences(text: str) -> list[str]:
    """Split into sentences. Crude but stable, which is what matters here."""
    parts = [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def count_syllables(word: str) -> int:
    """Heuristic English syllable count: vowel groups, minus silent trailing e."""
    lowered = word.lower().strip("'-")
    if not lowered:
        return 0
    groups = _VOWEL_GROUP.findall(lowered)
    count = len(groups)
    if lowered.endswith("e") and not lowered.endswith(("le", "ee", "ye")) and count > 1:
        count -= 1
    return max(1, count)


def flesch_reading_ease(text: str) -> float:
    """Standard Flesch Reading Ease. Higher is easier; 60-70 is plain English."""
    sentence_list = sentences(text)
    word_list = words(text)
    if not sentence_list or not word_list:
        return 0.0
    syllables = sum(count_syllables(w) for w in word_list)
    words_per_sentence = len(word_list) / len(sentence_list)
    syllables_per_word = syllables / len(word_list)
    return 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word


def count_phrases(text: str, phrases: tuple[str, ...]) -> int:
    lowered = text.lower()
    total = 0
    for phrase in phrases:
        total += len(re.findall(rf"\b{re.escape(phrase)}\b", lowered))
    return total


def compute_metrics(text: str) -> TextMetrics:
    """Measure a draft. Safe on empty input."""
    sentence_list = sentences(text)
    word_list = words(text)
    if not word_list:
        return TextMetrics()

    lengths = [len(words(s)) for s in sentence_list] or [0]
    return TextMetrics(
        word_count=len(word_list),
        sentence_count=len(sentence_list),
        avg_sentence_words=sum(lengths) / len(lengths),
        # Low variance reads as monotonous; the rubric penalises it, so measure it.
        sentence_length_stdev=statistics.pstdev(lengths) if len(lengths) > 1 else 0.0,
        longest_sentence_words=max(lengths),
        flesch_reading_ease=flesch_reading_ease(text),
        hedge_count=count_phrases(text, HEDGES),
        filler_count=count_phrases(text, FILLERS),
        passive_hits=len(_PASSIVE.findall(text)),
    )


def unified_diff_summary(before: str, after: str, *, max_hunks: int = 12) -> dict[str, object]:
    """Summarise what actually changed between two drafts.

    Exists to stop the agent grading its own homework. Without it, a revision
    that returned the text nearly unchanged would still be reported as "revised
    for evidence", and the loop would happily plateau while believing it was
    working.
    """
    import difflib

    before_lines = before.splitlines()
    after_lines = after.splitlines()
    matcher = difflib.SequenceMatcher(None, before_lines, after_lines, autojunk=False)
    similarity = matcher.ratio()

    added: list[str] = []
    removed: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            removed.extend(line.strip() for line in before_lines[i1:i2] if line.strip())
        if tag in ("replace", "insert"):
            added.extend(line.strip() for line in after_lines[j1:j2] if line.strip())

    before_words = len(words(before))
    after_words = len(words(after))
    return {
        "similarity": round(similarity, 4),
        "changed": similarity < 0.999,
        "lines_added": len(added),
        "lines_removed": len(removed),
        "word_delta": after_words - before_words,
        "words_before": before_words,
        "words_after": after_words,
        "added_sample": added[:max_hunks],
        "removed_sample": removed[:max_hunks],
    }


__all__ = [
    "FILLERS",
    "HEDGES",
    "compute_metrics",
    "count_phrases",
    "count_syllables",
    "flesch_reading_ease",
    "sentences",
    "unified_diff_summary",
    "words",
]
