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
# (you will probably need to import more things here)
import numpy as np
from regression import (logreg, utils)

def test_prediction():
	"""
	Check that prediction is working as expected
	"""
	model = logreg.LogisticRegressor(num_feats=2)
	model.W = np.array([1.0, -1.0, 0.5])

	X = np.array([
		[0.0, 0.0, 1.0],
		[2.0, 1.0, 1.0],
		[-1.0, 1.0, 1.0]
	])

	y_pred = model.make_prediction(X)
	expected = 1 / (1 + np.exp(-np.dot(X, model.W)))

	assert y_pred.shape == (3,)
	assert np.all((y_pred >= 0) & (y_pred <= 1))
	assert np.allclose(y_pred, expected)

def test_loss_function():
	"""
	Check that loss function is working as expected
	"""
	model = logreg.LogisticRegressor(num_feats=2)
	y_true = np.array([1.0, 0.0])
	y_pred = np.array([0.9, 0.2])

	loss = model.loss_function(y_true, y_pred)
	expected = -0.5 * (np.log(0.9) + np.log(0.8))

	assert loss == pytest.approx(expected)

def test_gradient():
	"""
	Check that gradient is working as expected
	"""
	model = logreg.LogisticRegressor(num_feats=2)
	model.W = np.array([0.2, -0.1, 0.3])

	X = np.array([
		[1.0, 2.0, 1.0],
		[0.5, -1.0, 1.0],
		[2.0, 0.0, 1.0]
	])
	y_true = np.array([1.0, 0.0, 1.0])

	grad = model.calculate_gradient(y_true, X)
	y_pred = model.make_prediction(X)
	expected = (X.T @ (y_pred - y_true)) / len(y_true)

	assert grad.shape == model.W.shape
	assert np.allclose(grad, expected)

def test_training():
	"""
	Check that training is working as expected
	"""
	model = logreg.LogisticRegressor(num_feats=2, learning_rate=0.1, max_iter=20, batch_size=2)

	X_train = np.array([
		[0.0, 0.0],
		[1.0, 1.0],
		[2.0, 2.0],
		[1.0, 0.0]
	])
	y_train = np.array([0.0, 1.0, 1.0, 0.0])

	X_val = np.array([
		[0.5, 0.5],
		[1.5, 1.0]
	])
	y_val = np.array([0.0, 1.0])

	initial_weights = model.W.copy()
	model.train_model(X_train, y_train, X_val, y_val)

	assert not np.allclose(initial_weights, model.W)
	assert len(model.loss_hist_train) > 0
	assert len(model.loss_hist_val) > 0