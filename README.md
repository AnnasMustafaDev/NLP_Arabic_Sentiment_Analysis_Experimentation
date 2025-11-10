# Sentiment Analysis and Named Entity Recognition on Arabic Tweets

## Abstract

This project addresses advanced sentiment analysis and named entity recognition (NER) for Arabic tweets related to the coronavirus pandemic. Employing modern natural language processing tools and machine learning models, this research analyzes public sentiments and identifies key entities in the Arabic social media landscape. The study leverages three traditional machine learning models and six state-of-the-art language models to provide comprehensive insights into public discourse surrounding COVID-19 in the Arab world.

## Table of Contents

- [Introduction](#introduction)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
  - [Tools and Libraries](#tools-and-libraries)
  - [Preprocessing](#preprocessing)
  - [Data Splitting](#data-splitting)
  - [Evaluation Metrics](#evaluation-metrics)
- [Model Development](#model-development)
  - [Machine Learning Models](#machine-learning-models)
  - [Large Language Models](#large-language-models)
  - [Named Entity Recognition](#named-entity-recognition)
- [Results](#results)
  - [Sentiment Analysis Performance](#sentiment-analysis-performance)
  - [Named Entity Recognition Results](#named-entity-recognition-results)
- [Conclusion](#conclusion)
- [Installation](#installation)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [References](#references)

## Introduction

The sentiment analysis and named entity recognition on Arabic tweets project develops a comprehensive system capable of analyzing sentiments expressed in Arabic tweets and identifying named entities within them. With the increasing popularity of social media platforms in the Arab world, understanding sentiment and extracting valuable information from tweets has become increasingly important.

This project leverages state-of-the-art natural language processing techniques and machine learning models to perform sentiment analysis and NER on Arabic tweets. By accurately identifying sentiments and extracting named entities, the system enables users to gain insights into public opinion, track trends, and understand sentiment associated with specific entities mentioned in tweets.

## Objectives

The primary objectives of this project are:

1. **Sentiment Analysis**: Utilize three machine learning models (Logistic Regression, Neural Networks, and Random Forest) to analyze sentiment expressed in Arabic tweets and classify them as positive or negative.

2. **Language Models Evaluation**: Explore the effectiveness of six state-of-the-art language models (BERT, AraBERT, DistilBERT, LLAMA2, GPT-3.5, and GPT-4) for sentiment analysis in Arabic language.

3. **Named Entity Recognition**: Implement NER techniques to identify and extract key entities mentioned in Arabic tweets, including locations, organizations, and individuals relevant to coronavirus pandemic discussions.

4. **Insights Generation**: Synthesize findings from sentiment analysis and NER to generate actionable insights that can inform decision-making processes, public health interventions, and communication strategies related to COVID-19.

## Dataset

**Dataset Name**: Arabic Corona Tweets Dataset

**Description**: The dataset contains 2400 Arabic language tweets related to the coronavirus pandemic.

**Structure**:

| Feature  | Description                                              | Data Type |
|----------|----------------------------------------------------------|-----------|
| Sentence | Contains 2400 Arabic tweet sentences                     | object    |
| Class    | Labels: 0 for negative sentiment, 2 for positive sentiment | int64     |

**Source**: Pre-annotated dataset labeled with sentiments for supervised learning.

## Methodology

### Tools and Libraries

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations and array operations
- **Matplotlib**: Data visualization and chart creation
- **Seaborn**: Enhanced data visualization capabilities
- **NLTK**: Natural language processing tasks (tokenization, stopwords removal)
- **WordCloud**: Generating word clouds for word frequency visualization
- **OpenAI**: Accessing GPT-3.5 and GPT-4 models
- **Transformers**: Accessing pre-trained language models (BERT, AraBERT, DistilBERT, LLAMA2)
- **Scikit-learn**: Machine learning tasks (text vectorization, classification)
- **Re**: Pattern matching and text processing
- **PrettyTable**: Generating formatted tables

### Preprocessing

Arabic language presents unique challenges for NLP due to its morphological complexities and dialectal varieties. The following preprocessing steps were implemented:

1. **Data Cleaning**: Removal of unnecessary characters, special characters, and symbols from tweet text.

2. **Normalization**: Standardization of Arabic character representation. Common characters such as "أ", "آ", and "إ" were normalized to "ا".

3. **Punctuation Removal**: Removal of Arabic and English punctuation marks using predefined lists.

4. **Tokenization**: Segmentation of cleaned tweet text into individual words using regular expressions.

5. **Stopwords Removal**: Elimination of common Arabic words (e.g., "و", "في", "على") to reduce noise and focus on meaningful content.

These preprocessing steps were applied uniformly to both positive and negative tweet datasets to prepare the text data for analysis.

### Data Splitting

The dataset was divided into training and testing segments:

- **Training Set**: 70% of the data for model training
- **Testing Set**: 30% of the data for model evaluation

This split ensures models are evaluated on previously unseen data, facilitating an equitable assessment of their predictive accuracy.

### Evaluation Metrics

Four quantitative metrics were used to assess model performance:

1. **Accuracy**: Ratio of correct predictions to total predictions
   - Formula: (TP + TN) / (TP + TN + FP + FN)

2. **Precision**: Correctness of positive predictions
   - Formula: TP / (TP + FP)

3. **Recall**: Proportion of correctly identified positive instances
   - Formula: TP / (TP + FN)

4. **F1-Score**: Harmonic mean of precision and recall
   - Formula: 2 × (Precision × Recall) / (Precision + Recall)

## Model Development

### Machine Learning Models

Three traditional machine learning algorithms were implemented:

1. **Logistic Regression**: A linear classification algorithm that estimates the probability of class membership using the logistic function. Chosen for its computational efficiency and interpretability.

2. **Random Forest**: An ensemble learning method based on decision trees. Constructs multiple decision trees during training and combines their predictions to improve accuracy and reduce overfitting.

3. **Artificial Neural Networks (ANN)**: Computational models inspired by the human brain structure, composed of interconnected layers of neurons. Excels at modeling complex, non-linear relationships within data.

### Large Language Models

Two categories of language models were employed:

#### Open-Source LLMs

- **BERT**: Bidirectional Encoder Representations from Transformers, renowned for understanding contextual information bidirectionally
- **AraBERT**: BERT variant specifically tailored for Arabic language
- **DistilBERT**: Distilled version of BERT, balancing inference speed and memory efficiency
- **LLAMA2**: Open-source language model designed for Arabic text processing

#### GPT Models

- **GPT-3.5**: Advanced generative model with strong contextual understanding
- **GPT-4**: Latest advancement in GPT series with improved architecture and performance

### Named Entity Recognition

NER implementation utilized the Abdusah/Arabert-ner model through the Hugging Face Transformers library. The process involved:

1. Tokenization of input text into manageable chunks
2. Processing each chunk through the NER pipeline
3. Identification and extraction of named entities (locations, organizations, individuals)
4. Handling of overlapping entities across text chunks
5. Consolidation of results for comprehensive entity recognition

## Results

### Sentiment Analysis Performance

#### Model Performance Summary

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 88.75%   | 92.21%    | 84.8%  | 88.37%   |
| Random Forest       | 86.5%    | 93%       | 79%    | 85.5%    |
| Neural Network      | 89%      | 93.82%    | 83.74% | 88.5%    |
| BERT                | 80%      | 80.9%     | 80%    | 79.85%   |
| DistilBERT          | 55.7%    | 64%       | 55.7%  | 47.9%    |
| AraBERT             | 82.1%    | 84%       | 82%    | 81.8%    |
| LLAMA2              | 70.7%    | 72.36%    | 70.7%  | 70%      |
| GPT-3.5             | 87%      | 88.2%     | 87.1%  | 87%      |
| GPT-4               | 92.85%   | 93.42%    | 92.85% | 92.83%   |

#### Key Findings

- **Best Performing Model**: GPT-4 achieved the highest accuracy at 92.85%
- **Top Traditional ML Model**: Neural Network demonstrated the highest accuracy among traditional models at 89%
- **Strong Performers**: Logistic Regression (88.75%) and GPT-3.5 (87%) showed robust performance
- **Arabic-Specific Models**: AraBERT (82.1%) outperformed generic BERT (80%) for Arabic text
- **Lowest Performer**: DistilBERT showed limited effectiveness for Arabic sentiment analysis at 55.7%

### Named Entity Recognition Results

The NER analysis revealed:

1. **Entity Distribution**: Most extracted entities were associated with negative sentiments due to the COVID-19 context
2. **Most Common Entity**: "كورونا" (Corona) appeared most frequently in the dataset
3. **Entity Categories**: Successfully identified locations, organizations, and individuals mentioned in tweets
4. **Sentiment Association**: Clear distinction between entities associated with positive and negative sentiments

WordCloud visualization demonstrated the frequency and prominence of recognized entities, with font size corresponding to occurrence frequency in the dataset.

## Conclusion

This comprehensive study evaluated sentiment analysis and Named Entity Recognition on Arabic tweets related to the coronavirus pandemic. Key achievements include:

1. **Model Performance**: GPT-4 achieved the highest accuracy (92.85%), followed by Neural Networks (89%) among traditional models.

2. **Arabic NLP Capabilities**: Successfully demonstrated effective sentiment classification and entity extraction from Arabic social media text.

3. **Practical Applications**: Results provide valuable insights for policymakers, healthcare professionals, and researchers addressing the COVID-19 pandemic in the Arab region.

4. **Future Directions**: Further research should focus on fine-tuning algorithms, incorporating dialectal variations, and expanding analysis scope to include regional differences.

The study underscores the importance of leveraging advanced NLP techniques for analyzing Arabic social media data, contributing to informed decision-making and effective crisis management strategies.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn nltk wordcloud
pip install transformers torch
pip install scikit-learn prettytable
pip install openai
```

### Additional Setup

For NLTK, download required data:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Usage

### Running the Notebook

1. Clone the repository:
```bash
git clone https://github.com/AnnasMustafaDev/NLP_Arabic_Sentiment_Analysis_Experimentation.git
cd NLP_Arabic_Sentiment_Analysis_Experimentation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open the Jupyter notebook:
```bash
jupyter notebook arabic-sentiment-analysis.ipynb
```

4. Execute cells sequentially to reproduce the analysis

### Using Individual Models

```python
# Example: Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize text
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)

# Predict
predictions = lr_model.predict(X_test_vec)
```

## Repository Structure

```
NLP_Arabic_Sentiment_Analysis_Experimentation/
├── arabic-sentiment-analysis.ipynb    # Main analysis notebook
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── data/                             # Dataset directory
│   └── arabic_corona_tweets.csv      # Arabic tweets dataset
├── models/                           # Saved model files
├── results/                          # Output results and visualizations
└── utils/                            # Helper functions and utilities
```

## References

1. Devlin, J., et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
2. Antoun, W., et al. (2020). AraBERT: Transformer-based Model for Arabic Language Understanding.
3. Sanh, V., et al. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter.
4. OpenAI. (2023). GPT-4 Technical Report.
5. Touvron, H., et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaboration opportunities, please contact:

- GitHub: [@AnnasMustafaDev](https://github.com/AnnasMustafaDev)
- Repository: [NLP_Arabic_Sentiment_Analysis_Experimentation](https://github.com/AnnasMustafaDev/NLP_Arabic_Sentiment_Analysis_Experimentation)

## Acknowledgments

This research was conducted as part of advanced natural language processing experimentation for Arabic text analysis. We acknowledge the contributions of the open-source community and the developers of the language models and libraries used in this project.
