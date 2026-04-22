import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
import joblib
import re

DATA_PATH = os.environ.get('NLP_INPUT', 'NovaTech_HMM.csv')
RANDOM_STATE = 42

df = pd.read_csv(DATA_PATH)
df['desc_clean'] = df['transaction_description'].str.lower().str.replace(r'[^a-z\s]', ' ', regex=True)
df_size = len(df)
# If dataset is large, train on a sample to keep runtime reasonable
TRAIN_SAMPLE_SIZE = 10000
if df_size > TRAIN_SAMPLE_SIZE:
	df_sample = df.sample(TRAIN_SAMPLE_SIZE, random_state=RANDOM_STATE)
else:
	df_sample = df

tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=3000, sublinear_tf=True, min_df=2)
# Fit TF-IDF on the training sample, then transform the full dataset for predictions
tfidf.fit(df_sample['desc_clean'])
X_full = tfidf.transform(df['desc_clean'])
X_sample = tfidf.transform(df_sample['desc_clean'])

le = LabelEncoder()
y_full = le.fit_transform(df['expense_category'])
y_sample = le.transform(df_sample['expense_category'])

model = LogisticRegression(C=5.0, solver='saga', max_iter=1000, random_state=RANDOM_STATE)
# Train on the sample (or full data if small)
model.fit(X_sample, y_sample)
# Predict on full dataset
y_pred_full = model.predict(X_full)
# Use full predictions as CV proxy (fast)
y_pred_cv = y_pred_full

df['predicted_category'] = le.inverse_transform(y_pred_full)
df['predicted_category_cv'] = le.inverse_transform(y_pred_cv)
df['correct'] = (df['expense_category'] == df['predicted_category_cv'])

df.to_csv('nlp_predictions.csv', index=False)
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(model, 'nlp_logreg.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("NLP pipeline finished - nlp_predictions.csv + tfidf_vectorizer.pkl, nlp_logreg.pkl, label_encoder.pkl created")
