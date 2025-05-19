# === IMPORTS ===
import pandas as pd
import numpy as np
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib
# === 1. Load Dataset ===
df = pd.read_csv('fake_job_postings.csv')  # Ensure this CSV is in the same folder
print("Dataset loaded. Shape:", df.shape)

# === 2. Basic Cleaning ===
df.drop(['telecommuting', 'has_company_logo', 'has_questions', 'salary_range'], axis=1, inplace=True)
df.dropna(subset=['description', 'requirements'], inplace=True)
df['text'] = df['description'] + " " + df['requirements']
df = df[['text', 'fraudulent']]  # Keep only needed columns

# === 3. Text Preprocessing ===
df['text'] = df['text'].str.lower().str.translate(str.maketrans('', '', string.punctuation))

# === 4. TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X = vectorizer.fit_transform(df['text'])
y = df['fraudulent']

# === 5. Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 6. Model Training ===
model = MultinomialNB()
model.fit(X_train, y_train)

# === 7. Model Prediction & Evaluation ===
y_pred = model.predict(X_test)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# === 8. Plot Confusion Matrix ===
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# === 9. Plot Class Distribution ===
plt.figure(figsize=(4, 3))
sns.countplot(x='fraudulent', data=df)
plt.title("Class Distribution")
plt.xlabel("Class (0 = Real, 1 = Fake)")
plt.ylabel("Number of Job Posts")
plt.tight_layout()
plt.show()

# === 10. Export Predictions to CSV (Optional but useful) ===
results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
results_df.to_csv("fake_job_predictions.csv", index=False)
print("\nPredictions saved to fake_job_predictions.csv")
joblib.dump(model, "naive_bayes_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")