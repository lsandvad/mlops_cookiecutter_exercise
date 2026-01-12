# from corrupted_mnist.data import corrupt_mnist
# from data import corrupt_mnist
# from model import MyModel
from src.cookiecutter_mlops_m6.data import corrupt_mnist
from src.cookiecutter_mlops_m6.model import MyAwesomeModel
import torch
import pytest
import re
import os.path


@pytest.mark.skipif(not os.path.exists("data"), reason="Data files not found")
def test_data():
    train, test = corrupt_mnist()
    assert len(train) == 30000
    assert len(test) == 5000
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0, 10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0, 10)).all()


@pytest.mark.parametrize("batch_size", [32, 64])
def test_model_with_parameters(batch_size: int) -> None:
    model = MyAwesomeModel()
    x = torch.randn(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)


def test_model():
    model = MyAwesomeModel()
    x = torch.randn(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)


def test_error_on_wrong_shape():
    model = MyAwesomeModel()
    with pytest.raises(ValueError, match="Expected input to a 4D tensor"):
        model(torch.randn(1, 2, 3))
    with pytest.raises(
        ValueError, match=re.escape("Expected each sample to have shape [1, 28, 28]")
    ):
        model(torch.randn(1, 1, 28, 29))
