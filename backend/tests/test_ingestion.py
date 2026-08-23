from backend.ingestion.document_parser import _guess_section
from backend.ingestion.sentence_segmenter import split_sentences


def test_split_sentences_basic():
    text = "This is one sentence. This is another! Is this a third?"
    sents = split_sentences(text)
    assert len(sents) == 3


def test_split_sentences_handles_abbreviations_reasonably():
    text = "Dr. Smith went to the store. He bought milk."
    sents = split_sentences(text)
    # regex fallback may or may not perfectly handle "Dr.", but should not
    # explode into many spurious fragments
    assert len(sents) <= 3


def test_guess_section_detects_heading():
    assert _guess_section("INTRODUCTION")
    assert _guess_section("1. Overview")
    assert not _guess_section("This is a normal sentence that ends with a period.")
    assert not _guess_section("")
