import pytest

from research_assistant.text.chunking import chunk_paper, heading, split_sections, strip_back_matter, window
from research_assistant.text.pdf import PdfError, clean_text, extract_text
from research_assistant.text.tokenize import (
    approx_tokens,
    normalize_for_match,
    split_paragraphs,
    split_sentences,
    tokenize,
)
from tests.pdf_helper import make_pdf


def test_tokenize_keeps_hyphenated_terms_and_drops_stopwords():
    assert tokenize("The Retrieval-Augmented model is used for QA, a task.") == ["retrieval-augmented", "model", "qa", "task"]


def test_approx_tokens():
    assert approx_tokens("one two three four five six seven eight nine ten") == 13
    assert approx_tokens("") == 0


def test_split_sentences_handles_abbreviations_and_decimals():
    text = "Jiang et al. (2023) report gains of 3.5 points. Results improve, e.g. on QA. Fig. 2 shows this! Next one?"
    assert split_sentences(text) == [
        "Jiang et al. (2023) report gains of 3.5 points.",
        "Results improve, e.g. on QA.",
        "Fig. 2 shows this!",
        "Next one?",
    ]


def test_split_paragraphs_and_normalize():
    assert split_paragraphs("a b\n\n  \n c d \n\ne") == ["a b", "c d", "e"]
    assert normalize_for_match("\u201cSmart\u201d  quotes\u2014and\u00a0dashes ") == '"smart" quotes-and dashes'


def test_heading_detection():
    assert heading("3 Method") == "Method"
    assert heading("2.1 Notations and Definitions") == "Notations and Definitions"
    assert heading("Related Work") == "Related Work"
    assert heading("2022. Improving language models by retrieving") is None  # a year in a reference list
    assert heading("1 Model 32.5 41.2 55.0 60.1") is None  # a table row
    assert heading("we embed the text") is None


def test_split_sections_skips_abstract_and_stops_at_references():
    text = "Title\nAbstract\nabstract text\n1 Introduction\nintro text\n2 Method\nmethod text\nReferences\n[1] ref"
    assert split_sections(text) == [("Introduction", "intro text"), ("Method", "method text")]
    assert strip_back_matter("a\nb\nAppendix\nc") == "a\nb"


def test_window_packs_whole_sentences_with_overlap():
    text = " ".join(f"Sentence number {i} has five words." for i in range(40))
    pieces = window(text, size_tokens=40, overlap_tokens=10)
    assert len(pieces) > 1
    assert all(p.endswith(".") for p in pieces)
    last_of_first = split_sentences(pieces[0])[-1]
    assert pieces[1].startswith(last_of_first)


def test_chunk_paper_abstract_first_then_sections_or_body():
    chunks = chunk_paper("We study RAG.", "1 Introduction\nRAG helps.\n2 Method\nWe retrieve.", 500, 50)
    assert [(c.ord, c.section, c.text) for c in chunks] == [
        (0, "Abstract", "We study RAG."), (1, "Introduction", "RAG helps."), (2, "Method", "We retrieve."),
    ]
    no_headings = chunk_paper("Abs.", "Plain text without headings.\nReferences\n[1] x", 500, 50)
    assert no_headings[1].section == "Body" and "[1]" not in no_headings[1].text
    assert chunk_paper("", None) == []


def test_pdf_extract_and_clean():
    pdf = make_pdf(["1 Introduction", "Retrieval helps when models hallu-", "cinate facts.", "References", "[1] x"])
    text = clean_text(extract_text(pdf))
    assert "hallucinate facts." in text
    assert clean_text("e\ufb03cient  and\u00adsoft") == "efficient andsoft"


def test_broken_pdf_raises_pdf_error():
    with pytest.raises(PdfError):
        extract_text(b"%PDF-1.4 this is not really a pdf")
