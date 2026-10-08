# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
import torch
from char_rnn_attention.data import TINY_TEXT, CharVocab, TinyCharDataset, make_tiny_dataset


def test_tiny_fixture_is_deterministic_and_offline() -> None:
    first = make_tiny_dataset(seq_len=16)
    second = make_tiny_dataset(seq_len=16)

    first_x, first_y = first[0]
    second_x, second_y = second[0]

    assert TINY_TEXT
    assert len(first) == len(second)
    assert torch.equal(first_x, second_x)
    assert torch.equal(first_y, second_y)
    assert first_x.shape == (16,)
    assert first_y.shape == (16,)
    assert first_x.dtype == torch.long
    assert first.vocab.decode(first_x.tolist()) == TINY_TEXT[:16]


def test_vocab_round_trip_and_dataset_bounds() -> None:
    vocab = CharVocab("aba")
    encoded = vocab.encode("baa")

    assert vocab.decode(encoded) == "baa"
    assert vocab.size == 2
    assert len(TinyCharDataset("aba", seq_len=1, vocab=vocab)) == 2


def test_factory_does_not_need_a_download() -> None:
    dataset = make_tiny_dataset(seq_len=8, repeats=2)

    assert len(dataset) > 0
    assert dataset.text == TINY_TEXT * 2
