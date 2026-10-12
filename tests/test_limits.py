"""Dataset size and label-length caps."""

from pipeline.label_length import LabelPiece, plan_pieces

from indic_ocr_pipeline.limits import line_quotas, pages_to_keep


def test_ten_lines_use_two_short_prefixes() -> None:
    assert pages_to_keep([3, 5, 85, 12], 10) == [2, 1]


def test_larger_budget_stops_inside_a_long_pdf() -> None:
    assert pages_to_keep([3, 5, 85], 100) == [3, 5, 17]


def test_labels_stay_within_ten_characters_and_mostly_under_five() -> None:
    text = "महाराष्ट्र विधान परिषदेच्या ०३ पदवीधर"
    pieces = plan_pieces([text])[0]
    lengths = [piece.length for piece in pieces]
    assert lengths
    assert max(lengths) <= 10
    assert all(length < 5 or 8 <= length <= 10 for length in lengths)
    assert sum(length < 5 for length in lengths) / len(lengths) >= 0.95
    assert "".join(piece.text for piece in pieces) == "".join(text.split())


def test_eleven_digits_are_split_and_digits_are_kept() -> None:
    digits = "१२३४५६७८९०१"
    pieces = plan_pieces([digits])[0]
    assert all(piece.length <= 10 for piece in pieces)
    assert "".join(piece.text for piece in pieces) == digits


def test_a_large_set_keeps_a_few_labels_of_eight_to_ten() -> None:
    planned = plan_pieces(["1234567890"] * 20)
    lengths = [piece.length for line in planned for piece in line]
    long = [length for length in lengths if length >= 8]
    assert long
    assert all(8 <= length <= 10 for length in long)
    assert sum(length < 5 for length in lengths) / len(lengths) >= 0.95


def test_two_digit_label_is_unchanged() -> None:
    assert plan_pieces(["०३"]) == [[LabelPiece("०३", 0, 2)]]


def test_longest_label_ignores_spaces() -> None:
    from pipeline.runners import _longest_label

    rows = [{"expected_text": "ab cd"}, {"gold_text": "xyz"}]
    assert _longest_label(rows) == 4


def test_line_budget_is_split_across_documents() -> None:
    assert line_quotas(10, 2) == [5, 5]
    assert sum(line_quotas(10, 3)) == 10
