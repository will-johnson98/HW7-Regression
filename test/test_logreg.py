"""
Write your unit tests here. Some tests to include are listed below.
This is not an exhaustive list.

- check that prediction is working correctly
- check that your loss function is being calculated correctly
- check that your gradient is being calculated correctly
- check that your weights update during training
"""

# Imports
import pytest
import numpy as np
from regression.logreg import LogisticRegressor

def test_prediction():
	"""
	Test that the sigmoid function is applied correctly.
	"""
	model = LogisticRegressor(num_feats=2)
	model.W = np.array([0.5, -0.5, 0]) # one for each feature and one bias term

	X = np.array([[1.0, 2.0],
			      [3.0, 4.0]])
	X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))]) # add bias to input

	z = np.dot(X_with_bias, model.W)
	expected = 1 / (1 + np.exp(-z))
	predictions = model.make_prediction(X_with_bias)

	assert np.allclose(predictions, expected)


def test_loss_function():
	"""
	Test that binary cross-entropy loss is calculated correctly.
	"""
	model = LogisticRegressor(num_feats=1)

	y_true = np.array([0, 1, 0, 1])
	y_pred = np.array([0.1, 0.9, 0.2, 0.8])

	# Expected Loss
	epsilon = 1e-15
	y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
	expected_loss = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

	loss = model.loss_function(y_true, y_pred)
	assert np.isclose(loss, expected_loss)


def test_gradient():
    """
    Test that gradient is calculated correctly.
    """
    model = LogisticRegressor(num_feats=1)
    model.W = np.array([0.5, 0.0])  # Weight and bias
    
    # Test data
    X = np.array([[1.0], [2.0]])
    X_with_bias = np.hstack([X, np.ones((X.shape[0], 1))])
    y_true = np.array([0, 1])
    
    # Calculate predictions and expected gradient
    y_pred = model.make_prediction(X_with_bias)
    expected_gradient = np.dot(X_with_bias.T, (y_pred - y_true)) / X.shape[0]
    
    # Test gradient calculation
    gradient = model.calculate_gradient(y_true, X_with_bias)
    assert np.allclose(gradient, expected_gradient)


def test_training():
    """
    Test that weights update during training.
    """
    model = LogisticRegressor(num_feats=1, learning_rate=0.1, 
                             max_iter=10, batch_size=1)
    
    # Save initial weights
    initial_weights = model.W.copy()
    
    # Simple dataset
    X_train = np.array([[1.0], [2.0]])
    y_train = np.array([0, 1])
    X_val = np.array([[1.5], [2.5]])
    y_val = np.array([0, 1])
    
    # Train the model
    model.train_model(X_train, y_train, X_val, y_val)
    
    # Verify weights changed and loss history populated
    assert not np.allclose(model.W, initial_weights)
    assert len(model.loss_hist_train) > 0
    assert len(model.loss_hist_val) > 0


def test_edge_cases():
    """Test model behavior with extreme values."""
    model = LogisticRegressor(num_feats=1)
    
    # Test near-zero and near-one predictions
    y_true = np.array([0, 1])
    y_pred_extreme = np.array([0.0001, 0.9999])
    loss = model.loss_function(y_true, y_pred_extreme)
    assert np.isfinite(loss)
    
    # Test very large input values (potential sigmoid overflow)
    X_large = np.array([[1000.0], [-1000.0]])
    X_large_with_bias = np.hstack([X_large, np.ones((X_large.shape[0], 1))])
    model.W = np.array([0.01, 0.0])
    
    predictions = model.make_prediction(X_large_with_bias)
    assert np.isfinite(predictions).all()
    assert np.all(predictions >= 0) and np.all(predictions <= 1)


def test_reset_model():
    """Test that reset_model correctly reinitializes the model."""
    model = LogisticRegressor(num_feats=1, learning_rate=0.1, 
                             max_iter=10, batch_size=1)
    
    # Train model
    X_train = np.array([[1.0], [2.0]])
    y_train = np.array([0, 1])
    X_val = np.array([[1.5], [2.5]])
    y_val = np.array([0, 1])
    model.train_model(X_train, y_train, X_val, y_val)
    
    # Save trained state
    trained_weights = model.W.copy()
    assert len(model.loss_hist_train) > 0
    
    # Reset and verify
    model.reset_model()
    assert not np.array_equal(model.W, trained_weights)
    assert len(model.loss_hist_train) == 0
    assert len(model.loss_hist_val) == 0