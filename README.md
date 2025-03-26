# Movie Review Sentiment Analysis

## Project Task
This project focuses on predicting the sentiment of movie reviews using Natural Language Processing (NLP). Specifically, we classify reviews from the IMDb Movie Review Stanford dataset as either positive (1) or negative (0) using a pre-trained language model.

## Dataset
The dataset consists of:
- **Training Data:** 25,000 labeled movie reviews with corresponding sentiment labels (0 for negative, 1 for positive).
- **Test Data:** 25,000 labeled reviews used for evaluation.
- **Unsupervised Data:** Additional unlabeled movie reviews available for potential use.
- **Data Type:** The dataset contains categorical text-based data, making it suitable for NLP tasks using pre-trained LLMs.

### Dataset Links
- **Unsupervised Data:** [unsupervised_clean.csv](https://drive.google.com/file/d/1tDVwWYhL6kEtZ8_2-ep3BmjTLt58dP0/view?usp=drive_link)
- **Training Data:** [clean_train.csv](https://drive.google.com/file/d/1MUsG2SpYJBucnLqs6tN4gNOR0WIV544Z/view?usp=drive_link)
- **Test Data:** [clean_test.csv](https://drive.google.com/file/d/1Y1TwntuzKpqdNQUxKqTPN2ZFnPf9XN-K/view?usp=drive_link)

## Pre-trained Model
The **DistilBERT** pre-trained model was selected for this task due to its efficiency and performance:
- **Why DistilBERT?**
  - Faster and more lightweight than BERT.
  - Requires fewer computational resources.
  - Maintains high accuracy while being more efficient.

## Performance Metrics
The model was evaluated using the following metrics:

| Metric       | Training Dataset | Test Dataset |
|-------------|----------------|--------------|
| Accuracy    | 89.7%          | 89.9%        |
| Precision   | 0.8977         | 0.8991       |
| Recall      | 0.8976         | 0.8991       |
| F1 Score    | 0.8976         | 0.8991       |
| Loss        | -              | 0.2705       |

![alt text](<Screenshot 2025-03-26 150706.png>)


## Hyperparameters
The following hyperparameters were found to be crucial in optimizing the model's performance:
- **Number of epochs:** Controls the number of times the model passes through the dataset.
- **Weight decay:** Helps prevent overfitting.
- **Warmup steps:** Stabilizes training at the beginning to prevent large weight updates.
- **Hidden dropout probability:** Prevents overfitting by randomly deactivating neurons.
- **Attention probabilities dropout probability:** Reduces reliance on specific tokens to improve generalization.

## Conclusion
This project successfully applies a DistilBERT-based model to predict IMDb movie review sentiments. By leveraging transfer learning, careful hyperparameter tuning, and dataset preprocessing, we achieved a strong classification performance with an accuracy of **89.9%** on unseen test data.

