# Model Evaluation

## Overfitting and Underfitting

**Overfitting** occurs when a model learns the noise in the training data and therefore performs well on training data but poorly on new data. **Underfitting** occurs when a model is too simple to capture the real pattern, so it performs poorly on both training and test data.

The **bias-variance tradeoff** describes the tension between the two failure modes. High bias means the model makes strong simplifying assumptions and underfits. High variance means the model is very sensitive to the particular training sample and overfits. Making a model more complex usually lowers bias but raises variance.

Common remedies for overfitting are collecting more data, using regularization, simplifying the model and stopping training early.

## Cross-Validation

**Cross-validation** estimates generalization performance by repeatedly training and evaluating on different splits of the data. In **k-fold cross-validation** the data is divided into k equal folds. Each fold is used once as the validation set while the remaining k - 1 folds are used for training, and the k scores are averaged.

Cross-validation gives a more stable estimate than a single split, especially for small datasets, but it costs k times as much training.

### Worked Example: Mean Cross-Validation Accuracy

**Problem:** In 5-fold cross-validation the fold accuracies are 0.80, 0.84, 0.78, 0.82 and 0.86. What is the mean accuracy?
**Solution:** The sum is 0.80 + 0.84 + 0.78 + 0.82 + 0.86 = 4.10, so the mean is 4.10 / 5 = 0.82.

## Classification Metrics

A **confusion matrix** counts true positives, false positives, true negatives and false negatives. **Precision** is the fraction of predicted positives that are truly positive. **Recall** is the fraction of truly positive examples that the model finds. The **F1 score** is the harmonic mean of precision and recall, so it is high only when both are high.

Accuracy can be misleading on imbalanced data. A model that always predicts the majority class can reach high accuracy while having zero recall for the minority class.

### Worked Example: Precision, Recall and F1

**Problem:** A classifier produces 40 true positives, 10 false positives and 20 false negatives. Compute precision, recall and F1.
**Solution:** Precision = 40 / (40 + 10) = 0.8. Recall = 40 / (40 + 20) = 0.67. F1 = 2 * 0.8 * 0.67 / (0.8 + 0.67) = 0.73.
