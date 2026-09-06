"""Package import smoke — no DistilGPT-2 download."""

from email_subject_generator import (
    OUTPUT_DIR,
    PROJECT_ROOT,
    clean_subject,
    naive_subject,
    pack_method_metrics,
)


def test_package_exports_and_portable_paths():
    assert PROJECT_ROOT.is_dir()
    assert OUTPUT_DIR.name == "outputs"
    assert (PROJECT_ROOT / "src" / "email_subject_generator").is_dir()
    assert clean_subject("Hello world update")
    assert naive_subject("The quarterly budget review is tomorrow")


def test_pack_method_metrics_import():
    scores = pack_method_metrics(["Budget Review"], ["Budget Review"])
    assert scores["rouge1"] > 0
