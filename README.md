# Movie Review Sentiment Analysis

## Project Task
This project focuses on predicting movie reviews using the IMDb movie review dataset from Stanford. The goal is to classify reviews as either positive (1) or negative (0) using a pre-trained language model.

## Dataset
The dataset consists of:
- 25,000 labeled training reviews
- 25,000 labeled test reviews
- Additional unlabeled data for potential use
- The data is categorical, containing text-based movie reviews and corresponding sentiment labels (0 for negative, 1 for positive).

### Loading the Original Dataset
You can load the original dataset using the `datasets` library:
```python
from datasets import load_dataset

ds = load_dataset("stanfordnlp/imdb")
```

### Preprocessed Dataset Links
The cleaned and preprocessed datasets are stored in Google Drive and can be accessed via the following links:
- **Unsupervised Data:** [unsupervised_clean.csv](https://drive.google.com/file/d/1tDVwWYhL6kEtZ8_2-ep3BmjTLt58dP0/view?usp=drive_link)
- **Training Data:** [clean_train.csv](https://drive.google.com/file/d/1MUsG2SpYJBucnLqs6tN4gNOR0WIV544Z/view?usp=drive_link)
- **Testing Data:** [clean_test.csv](https://drive.google.com/file/d/1Y1TwntuzKpqdNQUxKqTPN2ZFnPf9XN-K/view?usp=drive_link)

## Pre-trained Model
The **DistilBERT** pre-trained model was selected because it is a smaller, faster, and more efficient version of BERT, making it well-suited for NLP tasks while maintaining high accuracy.

## Performance Metrics
The model was evaluated using the following metrics:

| Metric      | Training Set | Test Set |
|------------|-------------|-------------|
| Accuracy   | 89.7%       | 89.9%       |
| Precision  | 0.8977      | 0.8991      |
| Recall     | 0.8976      | 0.8991      |
| F1 Score   | 0.8976      | 0.8990      |
| Loss       | -           | 0.2705      |

![alt text](<Screenshot 2025-03-26 150706-1.png>)

## Hyperparameters
The most relevant hyperparameters used for optimization include:
- **Number of epochs**: Determines the number of training cycles
- **Weight decay**: Helps prevent overfitting
- **Warmup steps**: Gradually increases learning rate for stability
- **Hidden dropout probability**: Reduces overfitting by randomly dropping connections
- **Attention dropout probability**: Prevents over-reliance on specific tokens

This project leverages transfer learning with fine-tuning to achieve high accuracy in sentiment analysis.

## Model Deployment
The trained model has been deployed on [Hugging Face](https://huggingface.co/jhovanisabuchi/distilbert-imdb-sentiment). You can access and test the model there.


