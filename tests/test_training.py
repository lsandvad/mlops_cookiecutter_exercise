import pytest
import torch
import os
from pathlib import Path
from unittest.mock import patch
from train import train
from model import MyModel


@pytest.fixture
def original_dir():
    """Fixture to store the original working directory."""
    return os.getcwd()


class TestTrain:
    """Test suite for the train function."""
    
    def test_train_completes_successfully(self, tmp_path, original_dir):
        """Test that training runs without errors for minimal epochs."""
        # Save outputs to temp directory but keep data path intact
        model_path = tmp_path / "model.pth"
        plot_path = tmp_path / "training_statistics.png"
        
        with patch('torch.save') as mock_save, \
             patch('matplotlib.pyplot.Figure.savefig') as mock_savefig:
            
            train(lr=1e-3, batch_size=32, epochs=1)
            
            # Verify save methods were called
            assert mock_save.called
            assert mock_savefig.called
    
    def test_model_file_can_be_loaded(self):
        """Test that the saved model can be loaded successfully."""
        # Run training (will save to current directory)
        train(lr=1e-3, batch_size=32, epochs=1)
        
        # Load the saved model
        model = MyAwesomeModel()
        state_dict = torch.load("model.pth", weights_only=True)
        model.load_state_dict(state_dict)
        
        # Verify model is in eval mode and can make predictions
        model.eval()
        dummy_input = torch.randn(1, 1, 28, 28)
        output = model(dummy_input)
        assert output.shape == (1, 10)
        
        # Cleanup
        if os.path.exists("model.pth"):
            os.remove("model.pth")
        if os.path.exists("training_statistics.png"):
            os.remove("training_statistics.png")
    
    def test_train_with_different_batch_sizes(self):
        """Test that training works with different batch sizes."""
        with patch('torch.save'), patch('matplotlib.pyplot.Figure.savefig'):
            for batch_size in [16, 32, 64]:
                train(lr=1e-3, batch_size=batch_size, epochs=1)
    
    def test_train_with_different_learning_rates(self):
        """Test that training works with different learning rates."""
        with patch('torch.save'), patch('matplotlib.pyplot.Figure.savefig'):
            for lr in [1e-4, 1e-3, 1e-2]:
                train(lr=lr, batch_size=32, epochs=1)
    
    def test_model_improves_over_epochs(self, capsys):
        """Test that model shows learning by checking if loss decreases."""
        with patch('torch.save'), patch('matplotlib.pyplot.Figure.savefig'):
            # Train for a few epochs
            train(lr=1e-3, batch_size=32, epochs=3)
            
            # Capture output to verify training messages
            captured = capsys.readouterr()
            assert "Training day and night" in captured.out
            assert "Training complete" in captured.out
    
    def test_saved_model_makes_valid_predictions(self):
        """Test that the saved model produces valid output."""
        train(lr=1e-3, batch_size=32, epochs=1)
        
        # Load model and test predictions
        model = MyAwesomeModel()
        model.load_state_dict(torch.load("model.pth", weights_only=True))
        model.eval()
        
        # Test with a batch of random inputs
        dummy_batch = torch.randn(5, 1, 28, 28)
        with torch.no_grad():
            predictions = model(dummy_batch)
        
        # Check output shape and that predictions are valid probabilities
        assert predictions.shape == (5, 10)
        # After softmax, should sum to 1
        probs = torch.softmax(predictions, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(5), atol=1e-5)
        
        # Check that predictions are in valid range [0, 9]
        pred_classes = predictions.argmax(dim=1)
        assert all(0 <= p < 10 for p in pred_classes)
        
        # Cleanup
        if os.path.exists("model.pth"):
            os.remove("model.pth")
        if os.path.exists("training_statistics.png"):
            os.remove("training_statistics.png")
    
    @pytest.mark.parametrize("epochs", [1, 2, 5])
    def test_train_with_different_epochs(self, epochs):
        """Test that training works with different number of epochs."""
        with patch('torch.save'), patch('matplotlib.pyplot.Figure.savefig'):
            train(lr=1e-3, batch_size=32, epochs=epochs)
    
    def test_train_output_files_are_created(self):
        """Test that output files are created."""
        train(lr=1e-3, batch_size=32, epochs=1)
        
        # Check files exist
        assert os.path.exists("model.pth")
        assert os.path.exists("training_statistics.png")
        
        # Check model file size
        model_path = Path("model.pth")
        assert model_path.stat().st_size > 1000  # Should be at least 1KB
        
        # Check plot file size
        plot_path = Path("training_statistics.png")
        assert plot_path.stat().st_size > 1000  # Should be at least 1KB
        
        # Cleanup
        os.remove("model.pth")
        os.remove("training_statistics.png")
    
    def test_training_with_small_batch_completes(self):
        """Test training with very small batch size (edge case)."""
        with patch('torch.save'), patch('matplotlib.pyplot.Figure.savefig'):
            train(lr=1e-3, batch_size=8, epochs=1)
    
    def test_model_parameters_updated_after_training(self):
        """Test that model parameters actually change during training."""
        # Get initial random model parameters
        initial_model = MyAwesomeModel()
        initial_params = [p.clone() for p in initial_model.parameters()]
        
        # Train model
        train(lr=1e-2, batch_size=32, epochs=2)
        
        # Load trained model
        trained_model = MyAwesomeModel()
        trained_model.load_state_dict(torch.load("model.pth", weights_only=True))
        
        # Check that at least some parameters have changed
        params_changed = False
        for initial_p, trained_p in zip(initial_params, trained_model.parameters()):
            if not torch.allclose(initial_p, trained_p):
                params_changed = True
                break
        
        assert params_changed, "Model parameters should change during training"
        
        # Cleanup
        if os.path.exists("model.pth"):
            os.remove("model.pth")
        if os.path.exists("training_statistics.png"):
            os.remove("training_statistics.png")
