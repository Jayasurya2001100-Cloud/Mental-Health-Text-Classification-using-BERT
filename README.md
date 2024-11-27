

# Mental Health Text Classification using HuggingFace BERT

This project demonstrates how to fine-tune a BERT model using the HuggingFace Transformers library for mental health text classification. It focuses on identifying mental health statuses such as anxiety and depression based on text inputs. Ideal for AI enthusiasts, data scientists, and mental health researchers, this project provides a comprehensive step-by-step guide.

## Overview
This project involves:
1. **Dataset Preparation**: Preprocessing raw text data, including cleaning and handling class imbalances.
2. **Model Fine-Tuning**: Using `BertForSequenceClassification` to fine-tune a pre-trained BERT model on labeled mental health data.
3. **Evaluation**: Generating metrics such as classification reports and confusion matrices.
4. **Deployment Preparation**: Saving the trained model and tokenizer for future use or deployment.

## Features
- **Data Cleaning**: Implements text cleaning using regular expressions and NLTK stopwords.
- **Imbalance Handling**: Resolves data imbalance using oversampling techniques (RandomOverSampler).
- **Model Training**: Fine-tunes BERT using the HuggingFace `Trainer` API.
- **Model Saving**: Saves the fine-tuned model, tokenizer, and label encoders for reuse.

## Installation
Ensure you have Python 3.7+ installed. Install the required dependencies using the command:
```bash
pip install -r requirements.txt
```

## Usage
### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/mental-health-text-classification.git
cd mental-health-text-classification
```

### 2. Open the Jupyter Notebook
Launch the notebook to follow the step-by-step guide:
```bash
jupyter notebook Mental_Health_Text_Classification_with_HuggingFace_BERT.ipynb
```

### 3. Dataset Preparation
Replace the placeholder dataset in the notebook with your own CSV file containing labeled mental health text data in the following format:
```csv
text,label
"I feel anxious about my exams",anxiety
"I have been feeling down lately",depression
```

### 4. Train the Model
Run the notebook cells to preprocess data, fine-tune the model, and evaluate its performance.

### 5. Save and Export the Model
The trained model, tokenizer, and label encoders are saved for future use. The notebook includes instructions to zip and download these files for deployment.

## Libraries Used
The project utilizes the following libraries:
- Pandas for data manipulation
- Scikit-learn for preprocessing and evaluation
- HuggingFace Transformers for BERT model and training utilities
- Imbalanced-learn for oversampling
- NLTK for text preprocessing
- Matplotlib and Seaborn for visualization
- PyTorch for backend computations

## Results
The notebook generates:
- **Evaluation Metrics**: Accuracy, precision, recall, F1 score.
- **Visualization**: Confusion matrices for performance insights.

