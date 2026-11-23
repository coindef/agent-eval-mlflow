import pytest

from agent_eval_mlflow.retrieval import BM25, CharNgramDense, rrf_fuse

DOCS = ["apple banana", "apple apple cherry", "cherry date"]


def test_bm25_score_matches_hand_computed_value():
    # N=3, df(banana)=1 -> idf = ln(1 + 2.5/1.5) = 0.98083; |d0|=2, avgdl=7/3, k1=1.5, b=0.75
    # score = 0.98083 * 2.5 / (1 + 1.5 * (0.25 + 0.75 * 2 / (7/3))) = 1.04822
    scores = BM25(DOCS).scores("banana")
    assert scores[0] == pytest.approx(1.04822, abs=1e-4)
    assert scores[1] == 0.0 and scores[2] == 0.0


def test_bm25_ranks_higher_term_frequency_first_and_drops_non_matches():
    ranking = BM25(DOCS).rank("apple")
    assert [i for i, _ in ranking] == [1, 0]


def test_bm25_ignores_stopwords():
    assert BM25(DOCS).rank("the of and") == []


def test_rrf_fuses_two_rankings_with_k60():
    # a: 1/61 + 1/62 = 0.032522, c: 1/63 + 1/61 = 0.032266, b: 1/62 = 0.016129
    fused = rrf_fuse([["a", "b", "c"], ["c", "a"]], k=60)
    assert [item for item, _ in fused] == ["a", "c", "b"]
    assert fused[0][1] == pytest.approx(0.032522, abs=1e-6)
    assert fused[2][1] == pytest.approx(0.016129, abs=1e-6)


def test_char_ngram_dense_matches_misspelled_query_that_bm25_misses():
    docs = ["Pallet Runner PR-200 warehouse robot", "SkyLift drone for inventory"]
    assert BM25(docs).rank("paletrunner") == []
    top = CharNgramDense(docs).rank("pallet runer")[0][0]
    assert top == 0
