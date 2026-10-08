# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
import torch
from char_rnn_attention.data import make_tiny_dataset
from char_rnn_attention.model import AdditiveAttention, CharGRUAttention, CharLSTMAttention


def test_additive_attention_returns_normalized_weights() -> None:
    attention = AdditiveAttention(hidden_size=8, attention_size=12)
    values = torch.randn(3, 7, 8)
    query = torch.randn(3, 5, 8)

    context, weights = attention(query, values)

    assert context.shape == (3, 5, 8)
    assert weights.shape == (3, 5, 7)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 5), atol=1e-6)
    assert torch.all(weights >= 0)


def test_attention_mask_excludes_padding_positions() -> None:
    attention = AdditiveAttention(hidden_size=4)
    values = torch.randn(2, 5, 4)
    query = torch.randn(2, 4, 4)
    mask = torch.tensor([[True, True, False, False, False], [True, True, True, False, False]])

    _, weights = attention(query, values, mask=mask)

    assert torch.allclose(
        weights.masked_select(~mask[:, None, :].expand_as(weights)),
        torch.zeros(20),
        atol=1e-7,
    )
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4), atol=1e-6)


def test_char_lstm_attention_preserves_sequence_shapes() -> None:
    dataset = make_tiny_dataset(seq_len=12)
    x, y = dataset[0]
    model = CharLSTMAttention(
        dataset.vocab.size,
        embedding_size=12,
        hidden_size=16,
        attention_size=20,
    )

    logits, weights = model(x.unsqueeze(0), y.unsqueeze(0))

    assert logits.shape == (1, 12, dataset.vocab.size)
    assert weights.shape == (1, 12, 12)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 12), atol=1e-6)


def test_char_gru_attention_preserves_sequence_shapes() -> None:
    dataset = make_tiny_dataset(seq_len=9)
    x, y = dataset[0]
    model = CharGRUAttention(dataset.vocab.size, embedding_size=8, hidden_size=12)

    logits, weights = model(x.unsqueeze(0), y.unsqueeze(0), teacher_forcing_ratio=0.0)

    assert logits.shape == (1, 9, dataset.vocab.size)
    assert weights.shape == (1, 9, 9)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(1, 9), atol=1e-6)


def test_char_lstm_attention_has_finite_gradients() -> None:
    torch.manual_seed(3)
    dataset = make_tiny_dataset(seq_len=10)
    x, y = dataset[0]
    model = CharLSTMAttention(dataset.vocab.size, embedding_size=8, hidden_size=12)

    logits, _ = model(x.unsqueeze(0))
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y)
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
