# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
import torch
from char_rnn_attention.data import make_tiny_dataset
from char_rnn_attention.model import CharLSTMAttention
from char_rnn_attention.train import fit, sequence_cross_entropy
from torch.utils.data import DataLoader


def test_sequence_cross_entropy_reduces_over_token_dimension() -> None:
    logits = torch.tensor([[[4.0, 0.0], [0.0, 4.0]]])
    targets = torch.tensor([[0, 1]])

    mean_loss = sequence_cross_entropy(logits, targets, reduction="mean")
    sum_loss = sequence_cross_entropy(logits, targets, reduction="sum")
    none_loss = sequence_cross_entropy(logits, targets, reduction="none")

    assert none_loss.shape == (1, 2)
    assert torch.allclose(sum_loss, none_loss.sum())
    assert torch.allclose(mean_loss, none_loss.mean())


def test_tiny_training_reduces_loss() -> None:
    torch.manual_seed(11)
    dataset = make_tiny_dataset(seq_len=14, repeats=4)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model = CharLSTMAttention(dataset.vocab.size, embedding_size=10, hidden_size=16)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    x, y = next(iter(loader))

    with torch.no_grad():
        initial = sequence_cross_entropy(model(x)[0], y).item()
    history = fit(model, loader, optimizer, epochs=3)

    assert len(history["loss"]) == 3
    assert history["loss"][-1] < initial
    assert all(torch.isfinite(torch.tensor(value)) for value in history["loss"])


def test_generate_is_deterministic_for_a_seeded_model() -> None:
    torch.manual_seed(5)
    dataset = make_tiny_dataset(seq_len=8)
    model = CharLSTMAttention(dataset.vocab.size, embedding_size=8, hidden_size=10)

    first = model.generate(dataset.vocab, "h", max_new_tokens=5, temperature=0.0)
    second = model.generate(dataset.vocab, "h", max_new_tokens=5, temperature=0.0)

    assert first == second
    assert first.startswith("h")
    assert len(first) == 6
