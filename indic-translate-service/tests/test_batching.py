import unittest

from translator import group_sentences, join_by_owner, length_batches


class BatchingTest(unittest.TestCase):
    def test_sentences_of_all_texts_are_regrouped_in_order(self):
        texts = ["पहिले वाक्य. दुसरे वाक्य.", "", "तिसरे वाक्य आहे."]
        sentences, owners = group_sentences(texts)
        assert sentences == ["पहिले वाक्य.", "दुसरे वाक्य.", "तिसरे वाक्य आहे."]
        assert owners == [0, 0, 2]
        english = ["first.", "second.", "third."]
        assert join_by_owner(english, owners, len(texts)) == ["first. second.", "", "third."]

    def test_batches_cover_every_sentence_once_shortest_first(self):
        sentences = ["a" * n for n in (30, 5, 20, 10, 25)]
        batches = length_batches(sentences, 2)
        assert sorted(i for b in batches for i in b) == [0, 1, 2, 3, 4]
        assert [len(sentences[i]) for i in batches[0]] == [5, 10]
        assert all(len(b) <= 2 for b in batches)


if __name__ == "__main__":
    unittest.main()
