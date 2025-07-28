# Decision Tree Analysis

## Focus of the Project

This project focuses on a specific **supervised learning** algorithm: the **Decision Tree Classifier**.

### What is Supervised Learning?

Supervised Learning (SL) is a subfield of machine learning where the model is trained on **labelled data**, meaning the input data is paired with the correct output. SL techniques are generally categorized into:
- **Classification**: Predicting discrete labels (e.g., spam vs. not spam)
- **Regression**: Predicting continuous values (e.g., housing prices)

In this study, we focus exclusively on **classification**, applying the decision tree algorithm to both:
- **Binary classification** problems
- **Multi-class classification** problems

---

## Why Decision Trees?

The **Decision Tree Classifier** is a widely used algorithm in supervised learning because:
- It requires minimal data preparation
- It is intuitive and easy to interpret
- It builds models by learning simple decision rules from the training data

This project implements the **ID3 (Iterative Dichotomiser 3)** algorithm to build decision trees from scratch.

---

## Dataset and Methodology

- A set of **classification datasets** (referenced in [1]) were used to train the model.
- These datasets include both **categorical** and **continuous** variables.
- Each dataset was split into a **training set** and a **test set**.
- The decision tree classifier (based on ID3) was trained on the training data and evaluated on the test data.

---

## Evaluation Metrics

Each model was evaluated using a standard **classification report**, including:
- **Precision**
- **Recall**
- **F1-Score**
- **Accuracy**

Further details on the implementation, datasets, and evaluation results are discussed in the following sections.
