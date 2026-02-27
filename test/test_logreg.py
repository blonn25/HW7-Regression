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

	# set known weights for testing and initialize X
	model.W = np.array([1.0, -1.0, 0.5])
	X = np.array([
		[0.0, 0.0, 1.0],
		[2.0, 1.0, 1.0],
		[-1.0, 1.0, 1.0]
	])

	# make predictions with the LogisticRegressor
	y_pred = model.make_prediction(X)

	# manually calculate expected predictions using the logistic function
	expected = 1 / (1 + np.exp(-np.dot(X, model.W)))

	# assert expected shape, range, and values of predictions
	assert y_pred.shape == (3,)
	assert np.all((y_pred >= 0) & (y_pred <= 1))
	assert np.allclose(y_pred, expected)

def test_loss_function():
	"""
	Check that loss function is working as expected
	"""
	model = logreg.LogisticRegressor(num_feats=2)

	# initialize easy to work with y_true and y_pred for testing
	y_true = np.array([1.0, 0.0])
	y_pred = np.array([0.9, 0.1])

	# compute loss using the LogisticRegressor
	loss = model.loss_function(y_true, y_pred)

	# manually calculate expected loss using the binary cross entropy formula 
	expected = -0.5 * (np.log(0.9) + np.log(0.9))

	# assert that the calculated loss is approximately equal to the expected loss
	assert loss == pytest.approx(expected)

def test_gradient():
	"""
	Check that gradient is working as expected
	"""
	model = logreg.LogisticRegressor(num_feats=2)

	# initialize know, easy to work with weights, X, and y_true
	model.W = np.array([0.2, -0.1, 0.3])
	X = np.array([
		[1.0, 2.0, 1.0],
		[0.5, -1.0, 1.0],
		[2.0, 0.0, 1.0]
	])
	y_true = np.array([1.0, 0.0, 1.0])

	# calculate the gradient using the LogisticRegressor
	grad = model.calculate_gradient(y_true, X)
	
	# manually calculate the expected gradient
	y_pred = model.make_prediction(X)
	expected = (X.T @ (y_pred - y_true)) / len(y_true)

	# assert that the calculated gradient has the same shape as the weights and is equal to the expected gradient
	assert grad.shape == model.W.shape
	assert np.allclose(grad, expected)

def test_training():
	"""
	Check that training is working as expected
	"""
	model = logreg.LogisticRegressor(num_feats=2, learning_rate=0.1, max_iter=20, batch_size=2)

	# initialize X_train, y_train with simple data for testing
	X_train = np.array([
		[0.0, 0.0],
		[1.0, 1.0],
		[2.0, 2.0],
		[1.0, 0.0]
	])
	y_train = np.array([0.0, 1.0, 1.0, 0.0])

	# initialize X_val, y_val with simple data for testing
	X_val = np.array([
		[0.5, 0.5],
		[1.5, 1.0]
	])
	y_val = np.array([0.0, 1.0])

	# save the initial weights and train the model
	initial_weights = model.W.copy()
	model.train_model(X_train, y_train, X_val, y_val)

	# assert that the weights have been updated and that the loss history lists are not empty
	assert not np.allclose(initial_weights, model.W)
	assert len(model.loss_hist_train) > 0
	assert len(model.loss_hist_val) > 0

def test_prediction_err():
	"""
	Check that an appropriate error is raised when prediction is misused
	"""
	model = logreg.LogisticRegressor(num_feats=2)

	# initialize X with 3 instances and 1 features (plus a bias term)
	X = np.array([
		[0.0, 0.0],
		[2.0, 1.0],
		[-1.0, 1.0]
	])

	# check that error is raised when X has wrong number of features (should have 2 feats based on how regrsessor was initialized)
	with pytest.raises(ValueError, match="X must have 3 features including bias term."):
		model.make_prediction(X)  # X has just 1 feature and 1 col for bias

def test_loss_function_err():
	"""
	Check that an appropriate error is raised when loss function is misused
	"""
	model = logreg.LogisticRegressor(num_feats=2)

	# initialize y_true with 3 instances and y_pred with 2 instances
	y_true = np.array([1.0, 0.0, 0.0])
	y_pred = np.array([0.9, 0.1])

	# check that error is raised when y_true and y_pred have different shapes
	with pytest.raises(ValueError, match="y_true and y_pred must be the same shape."):
		model.loss_function(y_true, y_pred)  # y_pred has one fewer element than y_true

def test_gradient_errs():
	"""
	Check that appropriate errors are raised when gradient is misused
	"""
	model = logreg.LogisticRegressor(num_feats=2)

	# initialize X with 3 instances and 2 features (plus bias term)
	X = np.array([
		[1.0, 2.0, 1.0],
		[0.5, -1.0, 1.0],
		[2.0, 0.0, 1.0]
	])

	# initialize y_true with 3 instances
	y_true = np.array([1.0, 0.0, 1.0])

	# check that error is raised when X has wrong number of features
	with pytest.raises(ValueError, match="X must have 3 features including bias term."):
		model.calculate_gradient(y_true, X[:, :-1])  # X truncated to just 1 feature and the bias term
	
	# check that error is raised when y_true and X are not compatible
	with pytest.raises(ValueError, match="y_true and X must have the same number of rows."):
		model.calculate_gradient(y_true[:-1], X)  # y_true truncatefd to just 2 elements, but X has 3 rows