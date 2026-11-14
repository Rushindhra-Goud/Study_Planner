# Machine Learning Foundations

## What Is Machine Learning

**Machine learning** is the study of algorithms that improve their performance on a task through experience with data instead of through explicitly programmed rules. A learning algorithm receives training data and produces a model that can make predictions on new, unseen inputs.

The three main paradigms are supervised learning, unsupervised learning and reinforcement learning. **Supervised learning** uses labeled examples, where each input is paired with the correct output. **Unsupervised learning** finds structure in unlabeled data, for example by grouping similar points together. **Reinforcement learning** trains an agent to choose actions that maximize cumulative reward.

## Supervised Learning Tasks

Supervised problems fall into two families. **Regression** predicts a continuous numeric value, such as the price of a house or tomorrow's temperature. **Classification** predicts a discrete category, such as whether an email is spam or not spam.

The input variables of a model are called **features**, and the quantity being predicted is called the **label** or target. Choosing informative features is often more important than choosing a complicated algorithm.

## Training, Validation and Test Data

A dataset is usually divided into three parts. The **training set** is used to fit the model parameters. The **validation set** is used to compare models and tune hyperparameters. The **test set** is used only once at the end to estimate how well the final model will perform on unseen data.

**Data leakage** occurs when information from the validation or test data influences training, which makes the measured performance look better than it really is. A common cause of leakage is computing normalization statistics on the full dataset before splitting it.

A typical split is 70 percent training, 15 percent validation and 15 percent test.

### Worked Example: Dataset Split

**Problem:** A dataset has 2000 examples and is split 70 percent for training, 15 percent for validation and 15 percent for testing. How many examples are in each part?
**Solution:** Training: 0.70 * 2000 = 1400 examples. Validation: 0.15 * 2000 = 300 examples. Test: 0.15 * 2000 = 300 examples. The three parts add up to 1400 + 300 + 300 = 2000 examples.
