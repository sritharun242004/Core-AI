import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def single_threaded_tiny_models():
    """Tiny CPU tensors run faster without thread-pool overhead."""
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)
