#Hello again welcome this is Yashwanth Reddy!!
#This is a task in ML internship provided by CODSOFT!!
#currently this is task -2 Spam SMS detection (April-2024)

# ML model for SMS spam detection using TF-IDF and classification algorithms (Naive Bayes, Logistic Regression, SVM)


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

data = pd.read_csv('C://Users//yaswa//OneDrive//Desktop//vscode//machine learning//spam.csv', encoding='latin-1')

data.drop_duplicates(inplace=True)
data['label'] = data['v1'].map({'ham': 'ham', 'spam': 'spam'})
X = data['v2']
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer()

X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

classifier = MultinomialNB()

classifier.fit(X_train_tfidf, y_train)

X_test_tfidf = tfidf_vectorizer.transform(X_test)

y_pred = classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average='micro')

recall = recall_score(y_test, y_pred, average='micro')

f1 = f1_score(y_test, y_pred, average='micro')

print('Progress: 100%')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-score: {f1:.2f}')

report = classification_report(y_test, y_pred, target_names=['Legitimate SMS', 'Spam SMS'])
print('Classification Report:')
print(report)
