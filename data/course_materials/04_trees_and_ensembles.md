# Trees and Ensembles

## Decision Trees

A **decision tree** predicts by asking a sequence of yes or no questions about the features, following one branch at each node until it reaches a leaf that holds the prediction. Trees are easy to interpret and need little data preparation, but a deep tree can overfit.

At each node the algorithm picks the split that makes the child nodes as pure as possible. **Entropy** and **Gini impurity** are two common ways to measure how mixed the classes in a node are. A node containing only one class has impurity zero.

**Information gain** is the entropy of the parent node minus the weighted average entropy of its children, and the tree chooses the split with the largest information gain.

### Worked Example: Gini Impurity

**Problem:** A node contains 8 samples, 4 positive and 4 negative. What is its Gini impurity?
**Solution:** The class proportions are 0.5 and 0.5. Gini = 1 - (0.5 * 0.5 + 0.5 * 0.5) = 0.5.

### Worked Example: Information Gain

**Problem:** A parent node has entropy 1.0. After a split, two children each hold half of the samples, with entropies 0.8 and 0.4. What is the information gain?
**Solution:** The weighted child entropy is 0.5 * 0.8 + 0.5 * 0.4 = 0.6. Information gain = 1.0 - 0.6 = 0.4.

## Random Forests

A **random forest** is an ensemble of many decision trees. Each tree is trained on a bootstrap sample of the data, which is a random sample drawn with replacement, and each split considers only a random subset of the features. The forest predicts by majority vote for classification or by averaging for regression.

Averaging many different trees reduces variance without increasing bias much, which is why a random forest overfits far less than a single deep tree.

### Worked Example: Majority Vote

**Problem:** A forest of 5 trees predicts the classes A, B, A, A and B for one sample. What does the forest output?
**Solution:** Class A receives 3 votes and class B receives 2 votes, so the forest predicts class A.

## Boosting

**Boosting** builds an ensemble sequentially, where each new model focuses on the examples that the previous models got wrong. **Gradient boosting** fits each new tree to the residual errors of the current ensemble. Boosting mainly reduces bias, but it can overfit if too many rounds are used, so the number of trees and the learning rate must be tuned.
