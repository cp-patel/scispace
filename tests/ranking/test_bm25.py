import numpy as np
import pytest

from research_assistant.ranking.bm25 import FieldSpec, bm25_scores, bm25f_scores, idf

FIELDS = {"title": FieldSpec(weight=2.0, b=0.3), "abstract": FieldSpec(weight=1.0, b=0.75)}


def doc(title: str, abstract: str) -> dict[str, list[str]]:
    return {"title": title.split(), "abstract": abstract.split()}


def test_idf_rarer_terms_weigh_more_and_never_negative():
    weights = idf(100, np.array([1, 10, 50, 100]))
    assert all(np.diff(weights) < 0)
    assert weights.min() > 0


def test_more_occurrences_score_higher_at_equal_length():
    docs = [doc("x", "rag a b c d"), doc("x", "rag rag b c d"), doc("x", "rag rag rag c d")]
    scores = bm25f_scores(docs, ["rag"], FIELDS)
    assert scores[0] < scores[1] < scores[2]


def test_term_frequency_saturates():
    def score(tf: int) -> float:
        docs = [doc("x", " ".join(["rag"] * tf + ["pad"] * (60 - tf))), doc("x", "other words here")]
        return float(bm25f_scores(docs, ["rag"], FIELDS)[0])

    assert score(2) - score(1) > score(50) - score(49)


def test_score_is_bounded_by_idf_times_k1_plus_one():
    docs = [doc("rag " * 30, "rag " * 300), doc("x", "y")]
    k1 = 1.2
    bound = idf(2, np.array([1]))[0] * (k1 + 1)
    assert bm25f_scores(docs, ["rag"], FIELDS, k1=k1)[0] < bound


def test_longer_abstract_scores_lower_for_same_count():
    docs = [doc("x", "rag " + "pad " * 5), doc("x", "rag " + "pad " * 200), doc("x", "none")]
    scores = bm25f_scores(docs, ["rag"], FIELDS)
    assert scores[0] > scores[1]


def test_title_hit_beats_abstract_hit():
    docs = [doc("rag systems", "a b c d e"), doc("other things", "rag b c d e"), doc("z", "z")]
    scores = bm25f_scores(docs, ["rag"], FIELDS)
    assert scores[0] > scores[1]


def test_empty_inputs_and_repeated_query_terms():
    assert bm25f_scores([], ["rag"], FIELDS).shape == (0,)
    docs = [doc("rag", "x"), doc("y", "z")]
    assert np.allclose(bm25f_scores(docs, ["rag", "rag"], FIELDS), bm25f_scores(docs, ["rag"], FIELDS))
    assert np.allclose(bm25f_scores(docs, [], FIELDS), 0.0)


def test_plain_bm25_matches_single_field_bm25f():
    docs = [["dense", "retrieval"], ["sparse", "retrieval", "bm25"], ["graphs"]]
    expected = bm25f_scores([{"text": d} for d in docs], ["retrieval", "bm25"], {"text": FieldSpec(1.0, 0.75)})
    assert np.allclose(bm25_scores(docs, ["retrieval", "bm25"]), expected)
    assert bm25_scores(docs, ["retrieval", "bm25"])[1] == pytest.approx(max(expected))
