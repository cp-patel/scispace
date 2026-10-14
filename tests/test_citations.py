from research_assistant.citations import (
    bibtex_entry,
    card_id,
    find_markers,
    has_malformed_markers,
    normalize_marker_placement,
    number_papers,
    paper_label,
    quote_in_source,
    reference_entry,
    render_markers,
)


def test_markers_parse_and_move_before_the_period():
    assert card_id(0, 3) == "E1.3"
    assert find_markers("A [E1.2, E3.1] and B [ E2.4 ].") == ["E1.2", "E3.1", "E2.4"]
    assert find_markers("Not a marker [E12] or [1].") == []
    assert normalize_marker_placement("Claim one. [E1.2] Claim two.") == "Claim one [E1.2]. Claim two."
    assert has_malformed_markers("Bad [E3] and [E2.x].") and not has_malformed_markers("Good [E1.2, E3.4].")


def test_quote_matching_ignores_case_spacing_and_end_punctuation():
    source = "Retrieval augmented models reduce   hallucination by 20%. They also cost more."
    assert quote_in_source("retrieval augmented models reduce hallucination by 20%", source)
    assert quote_in_source("They also cost more", source)
    assert not quote_in_source("They also cost less", source)
    assert not quote_in_source("   ", source)


def test_labels_numbering_and_rendering():
    assert paper_label(["Zhengbao Jiang", "Frank Xu", "Luyu Gao"], 2023) == "Jiang et al. 2023"
    assert paper_label(["Zhengbao Jiang", "Frank Xu"], 2023) == "Jiang and Xu 2023"
    assert paper_label([], None) == "Unknown n.d."
    card_paper = {"E1.1": "W1", "E1.2": "W2", "E2.1": "W1"}
    texts = ["Intro [E1.2].", "Body [E2.1, E1.1] and junk [E9.9]."]
    numbers = number_papers(texts, card_paper)
    assert numbers == {"W2": 1, "W1": 2}
    assert render_markers(texts[1], card_paper, numbers) == "Body [2] and junk."


def test_reference_and_bibtex_handle_missing_and_special_fields():
    paper = {"id": "W1", "title": "Costs & benefits of 100% recall_{x}", "authors": ["Ada Lovelace"], "year": 2020,
             "venue": "J. Stuff", "doi": "10.1/abc"}
    assert reference_entry(1, paper) == "[1] Ada Lovelace (2020). Costs & benefits of 100% recall_{x}. J. Stuff. https://doi.org/10.1/abc"
    bib = bibtex_entry(paper)
    assert bib.startswith("@article{lovelace2020costs,")
    assert r"Costs \& benefits of 100\% recall\_\{x\}" in bib
    assert "[1] Unknown authors (n.d.). Untitled." == reference_entry(1, {"id": "W2"})
    assert bibtex_entry({"id": "W2"}).startswith("@article{anonpaper,")
