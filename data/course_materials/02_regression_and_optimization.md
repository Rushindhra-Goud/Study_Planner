# Regression and Optimization

## Linear Regression

**Linear regression** models the label as a weighted sum of the features plus a bias term. With a single feature x the model is y = w * x + b, where w is the weight (slope) and b is the bias (intercept).

The standard loss function is the **mean squared error**, which averages the squared difference between each prediction and the true value. Squaring the errors makes every error positive and penalizes large mistakes more heavily than small ones.

### Worked Example: Mean Squared Error

**Problem:** A model predicts the values 3, 5 and 7 for true values 2, 5 and 9. Compute the mean squared error.
**Solution:** The errors (prediction minus truth) are 1, 0 and -2. The squared errors are 1, 0 and 4. MSE = (1 + 0 + 4) / 3 = 1.67.

### Worked Example: Making a Prediction

**Problem:** A fitted line is y = 2x + 1. What does it predict for x = 3?
**Solution:** Substitute x = 3 into the line. y = 2 * 3 + 1 = 7.

## Gradient Descent

**Gradient descent** is an iterative optimization method that repeatedly moves the parameters in the direction opposite to the gradient of the loss. The update rule is w_new = w - learning_rate * gradient.

The **learning rate** controls the size of each step. If the learning rate is too small, training is slow. If it is too large, the updates can overshoot the minimum and the loss may diverge.

**Stochastic gradient descent** computes the gradient from a single example or a small mini-batch rather than from the full dataset, which makes each update cheaper but noisier.

### Worked Example: One Gradient Descent Step

**Problem:** For the loss f(w) = w * w with learning rate 0.1 and starting point w = 4, compute w after one gradient descent step.
**Solution:** The gradient of f is 2 * w, which equals 8 at w = 4. The update is w_new = 4 - 0.1 * 8 = 3.2.

## Regularization

**Regularization** adds a penalty on the size of the weights to the loss so that the model prefers simpler solutions and overfits less. **Ridge regression** adds the sum of the squared weights, which is called the L2 penalty, multiplied by a strength parameter lambda. **Lasso regression** adds the sum of the absolute values of the weights, which is called the L1 penalty, and it can drive some weights exactly to zero.

A larger lambda means stronger regularization and a simpler model, while lambda equal to zero removes the penalty entirely.

### Worked Example: L2 Penalty

**Problem:** A model has weights 3 and -4 and regularization strength lambda = 0.5. Compute the L2 penalty lambda times the sum of squared weights.
**Solution:** The squared weights are 9 and 16, so the sum is 9 + 16 = 25. The penalty is 0.5 * 25 = 12.5.
