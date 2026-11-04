import unittest

from number_guard import fix_numbers
from text import MAX_WORDS_PER_PIECE, split_sentences, trim_repeats


class NumberGuardTest(unittest.TestCase):
    def test_dropped_number_is_appended(self):
        assert fix_numbers("अंदाजे रु. 11,50,000 खर्चून", "at an estimated cost of Rs.") == (
            "at an estimated cost of Rs. [11,50,000]"
        )

    def test_mangled_placeholder_is_replaced(self):
        assert fix_numbers("एफआयआर क्र. 0234/2022", "FIR No. < Id1 >") == "FIR No. 0234/2022"

    def test_reformatted_number_gets_source_format(self):
        assert fix_numbers("एकूण क्षेत्र 1.20 हेक्टर", "Total area: 1.2 ha") == "Total area: 1.20 ha"

    def test_mangled_currency(self):
        assert fix_numbers("किंमत ₹48,50,000", "price of ₹1") == "price of ₹48,50,000"

    def test_indic_digits_become_ascii(self):
        assert fix_numbers("क्षेत्र ५७.५ आर", "area R") == "area R [57.5]"

    def test_no_numbers_leaves_text_alone(self):
        assert fix_numbers("सदर मिळकत", "the said property") == "the said property"


class TextTest(unittest.TestCase):
    def test_splits_on_danda_and_full_stop_and_joins_line_wraps(self):
        text = "सदर मिळकत श्री. रमेश\nयांच्या नावे आहे। दुसरे वाक्य आहे."
        assert split_sentences(text) == ["सदर मिळकत श्री.", "रमेश यांच्या नावे आहे।", "दुसरे वाक्य आहे."]

    def test_long_unpunctuated_text_is_cut_into_pieces(self):
        words = [f"शब्द{i}" for i in range(MAX_WORDS_PER_PIECE * 2 + 5)]
        pieces = split_sentences(" ".join(words))
        assert [len(p.split()) for p in pieces] == [MAX_WORDS_PER_PIECE, MAX_WORDS_PER_PIECE, 5]

    def test_drops_fragments_without_letters(self):
        assert split_sentences("- . १") == []

    def test_trim_repeats(self):
        assert trim_repeats("ohhhhhhh no no no no") == "ohhh no"


if __name__ == "__main__":
    unittest.main()
