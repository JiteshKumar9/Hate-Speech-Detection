# Hate Speech Detection Using Machine Learning

This project implements a Hate Speech Detection system using machine learning techniques. The dataset contains tweets labeled as "hate speech," "offensive language," or "no hate or offensive language." The system preprocesses the text data, trains a Decision Tree Classifier, and evaluates its performance.

## Features
- Preprocessing of text data (removal of URLs, punctuation, stopwords, etc.)
- Stemming of words for feature reduction
- Vectorization of text data using CountVectorizer
- Model training using Decision Tree Classifier
- Evaluation of model performance using confusion matrix and accuracy score

## Dataset
The dataset used for this project is `twitter.csv`, which contains the following columns:
- `tweet`: The text of the tweet.
- `class`: The class label (0: hate speech, 1: offensive language, 2: no hate or offensive language).

The dataset is processed to include an additional column `labels`, mapping the class values to their respective descriptions.

## Prerequisites
- Python 3.7+
- Required libraries:
  - `pandas`
  - `numpy`
  - `nltk`
  - `sklearn`
  - `seaborn`
  - `matplotlib`

Install the necessary libraries using:
```bash
pip install pandas numpy nltk scikit-learn seaborn matplotlib
```

## Steps to Run the Project

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. **Dataset**:
   Ensure the `twitter.csv` file is in the root directory of the project.

3. **Run the Script**:
   Execute the Python script:
   ```bash
   python hate_speech_detection.py
   ```

4. **Sample Prediction**:
   Modify the `sample` variable in the script to test custom inputs. For example:
   ```python
   sample = "let's unite and kill all the people who are protesting against the government"
   ```

   The script will output the predicted label for the sample.

## Project Workflow

1. **Data Loading**:
   - Load the `twitter.csv` dataset using pandas.

2. **Data Cleaning**:
   - Convert text to lowercase.
   - Remove URLs, punctuation, HTML tags, numbers, and stopwords.
   - Apply stemming to reduce words to their base forms.

3. **Feature Extraction**:
   - Use `CountVectorizer` to transform text data into numerical features.

4. **Model Training**:
   - Split the data into training and testing sets.
   - Train a Decision Tree Classifier on the training data.

5. **Evaluation**:
   - Evaluate the model using a confusion matrix and accuracy score.
   - Visualize the confusion matrix using a heatmap.

6. **Prediction**:
   - Predict the class of a sample tweet.

## Key Functions

### Data Cleaning
The `clean_data` function preprocesses the text by performing the following steps:
- Lowercasing
- Removing URLs, punctuation, and stopwords
- Stemming words

### Model Training
The `DecisionTreeClassifier` from `sklearn` is used for training the model on vectorized text data.

### Evaluation
- **Confusion Matrix**: Displays the performance of the model.
- **Accuracy Score**: Measures the percentage of correctly classified samples.

## Results
The model achieves an accuracy of `X%` (replace with actual accuracy) on the test data.

## Example
### Input:
```python
sample = "let's unite and kill all the people who are protesting against the government"
```

### Output:
```python
Predicted Label: "hate speech"
```

## Visualizations
- Confusion matrix heatmap for model performance visualization.

## Future Improvements
- Use more advanced models like Random Forest, SVM, or Neural Networks.
- Experiment with TF-IDF vectorization for feature extraction.
- Include more preprocessing steps like lemmatization.
- Use a larger and more diverse dataset for better generalization.


