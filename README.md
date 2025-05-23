# Fake Job Post Detection

This project is a machine learning-based web application that detects potentially fake job postings. It leverages natural language processing (NLP) techniques to analyze job descriptions and requirements, predicting whether a posting is real or fraudulent.

## 📁 Project Structure

```
.
├── app.py                     # Streamlit web app for real-time predictions
├── fake_job_postings.py       # Script for preprocessing, training, and evaluation
├── fake_job_postings.csv      # Dataset of job postings
├── fake_job_predictions.csv   # Predictions on the test set
├── naive_bayes_model.pkl      # Trained Naive Bayes model (saved)
├── tfidf_vectorizer.pkl       # TF-IDF vectorizer (saved)
├── README.md                  # Project documentation
```

## 🔍 Features

* Cleans and preprocesses job data
* Vectorizes text using TF-IDF
* Trains a Naive Bayes classifier
* Evaluates performance with metrics and visualizations
* Provides an interactive UI with Streamlit for prediction

## 📊 Dataset

The `fake_job_postings.csv` dataset includes:

* `description`: The job description
* `requirements`: Skills and qualifications
* `fraudulent`: Label (0 = Real, 1 = Fake)

## 🧠 Model Training

The model is trained using a Multinomial Naive Bayes algorithm after vectorizing text inputs using TF-IDF.

To train the model and generate evaluation results:

```bash
python fake_job_postings.py
```

This script performs the following:

* Loads and cleans data
* Trains and evaluates the model
* Saves model and vectorizer
* Outputs predictions to `fake_job_predictions.csv`

## 🌐 Web Application

The Streamlit application (`app.py`) enables users to enter job details and receive a prediction in real time.

To run the app:

```bash
streamlit run app.py
```

### Inputs:

* Job Description
* Job Requirements

### Output:

* Prediction: Real or Fake
* Probabilities of each class

## 🛠️ Requirements

Install required packages with:

```bash
pip install -r requirements.txt
```

Key libraries:

* streamlit
* scikit-learn
* pandas
* numpy
* matplotlib
* seaborn
* joblib

## 📊 Visualizations

During training, the script also generates:

* Confusion matrix heatmap
* Class distribution bar chart
