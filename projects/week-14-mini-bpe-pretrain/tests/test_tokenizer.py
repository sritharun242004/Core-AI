import json

import pytest
from mini_bpe_pretrain import BPETokenizer

CORPUS = [
    "the tiny cat sat.",
    "the tiny dog sat.",
    "a tiny cat ran.",
]


def test_bpe_training_is_deterministic_and_merges_repeated_pairs() -> None:
    first = BPETokenizer.train(CORPUS, vocab_size=32)
    second = BPETokenizer.train(CORPUS, vocab_size=32)

    assert first.to_dict() == second.to_dict()
    assert first.merges
    assert first.vocab_size <= 32
    assert first.encode("the tiny cat") == second.encode("the tiny cat")


def test_encode_decode_round_trip_and_special_tokens() -> None:
    tokenizer = BPETokenizer.train(CORPUS, vocab_size=40)
    ids = tokenizer.encode("<bos>the tiny cat<eos>")

    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids) == "<bos>the tiny cat<eos>"
    assert tokenizer.decode(ids, skip_special_tokens=True) == "the tiny cat"


def test_unknown_text_uses_unk_and_json_round_trip(tmp_path) -> None:
    tokenizer = BPETokenizer.train(CORPUS, vocab_size=32)
    unknown_ids = tokenizer.encode("snow")
    assert tokenizer.unk_id in unknown_ids

    path = tmp_path / "tokenizer.json"
    tokenizer.save(path)
    restored = BPETokenizer.load(path)
    assert restored.to_dict() == tokenizer.to_dict()
    assert restored.decode(restored.encode("the tiny")) == "the tiny"
    assert json.loads(path.read_text())["merges"]


def test_invalid_tokenizer_inputs_are_explicit() -> None:
    with pytest.raises(ValueError):
        BPETokenizer.train([], vocab_size=12)
    with pytest.raises(ValueError):
        BPETokenizer.train(CORPUS, vocab_size=3)
    tokenizer = BPETokenizer.train(CORPUS, vocab_size=32)
    with pytest.raises(ValueError):
        tokenizer.decode([999])
