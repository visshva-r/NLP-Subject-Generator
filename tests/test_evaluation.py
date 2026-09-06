"""ROUGE wrapper and summary-stat tests (no model download)."""

from email_subject_generator.evaluation import compute_rouge, pack_method_metrics, summarize_metrics


def test_compute_rouge_returns_expected_keys():
    preds = ["Quarterly Budget Review", "Team Offsite Agenda"]
    refs = ["Quarterly Budget Review", "Offsite Agenda"]
    scores = compute_rouge(preds, refs)
    assert set(scores) >= {"rouge1", "rouge2", "rougeL"}
    for value in scores.values():
        assert 0.0 <= value <= 1.0
    assert scores["rouge1"] > 0.0


def test_summarize_metrics_length_and_uniqueness():
    generated = ["Alpha Beta", "Alpha Beta", "Gamma Delta Echo"]
    summary = summarize_metrics(generated, rouge={"rouge1": 0.1, "rouge2": 0.0, "rougeL": 0.1})
    assert summary["n"] == 3
    assert summary["unique_ratio"] == 2 / 3
    assert summary["avg_subject_length"] > 0
    assert summary["rouge1"] == 0.1


def test_pack_method_metrics_has_comparison_keys():
    packed = pack_method_metrics(
        ["Quarterly Budget Review", "Team Offsite"],
        ["Quarterly Budget Review", "Offsite Agenda"],
    )
    assert set(packed) >= {
        "rouge1",
        "rouge2",
        "rougeL",
        "n",
        "avg_subject_length",
        "unique_ratio",
    }
    assert packed["n"] == 2
    assert packed["rouge1"] > 0
