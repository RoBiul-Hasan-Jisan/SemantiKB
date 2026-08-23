from backend.evaluation.metrics import ndcg_at_k, precision_at_k, reciprocal_rank, recall_at_k


def test_precision_at_k():
    retrieved = ["a", "b", "c", "d"]
    relevant = {"a", "c"}
    assert precision_at_k(retrieved, relevant, 4) == 0.5
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert precision_at_k([], relevant, 4) == 0.0


def test_recall_at_k():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "c", "z"}
    assert round(recall_at_k(retrieved, relevant, 3), 4) == round(2 / 3, 4)


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == 1 / 3
    assert reciprocal_rank(["a"], {"a"}) == 1.0
    assert reciprocal_rank(["x"], {"a"}) == 0.0


def test_ndcg_perfect_order():
    retrieved = ["a", "b", "c"]
    relevant = {"a", "b"}
    score = ndcg_at_k(retrieved, relevant, 3)
    assert 0.9 <= score <= 1.0
