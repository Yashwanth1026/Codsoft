#Hello welcome this is Yashwanth Reddy!!
#This is a task in ML internship provided by CODSOFT!!
#currently this is task -1 movie generation (April-2024)

#The below code provides the predicted data in model_evalution file by providing the training data of movies along with bargraphs after prediction


import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

genre_list = ['animation', 'biography', 'comedy', 'crime', 'documentary', 'family', 'fantasy', 'game-show', 'history', 'horror', 'music','action', 'adult', 'adventure', 'short', 'sport', 'talk-show', 'thriller', 'war', 'western', 'musical', 'mystery', 'news', 'reality-tv', 'romance', 'sci-fi']

fallback_genre = 'Unknown'

try:
    with tqdm(total=50, desc="Loading Train Data") as pbar:
        train_data = pd.read_csv('C://Users//yaswa//OneDrive//Desktop//vscode//CodSoft//train_data.txt', sep=':::', header=None, names=['SerialNumber', 'MOVIE_NAME', 'GENRE', 'MOVIE_PLOT'], engine='python')
        pbar.update(50)
except Exception as e:
    print(f"Error loading train_data: {e}")

X_train = train_data['MOVIE_PLOT'].astype(str).apply(lambda doc: doc.lower())
genre_labels = [genre.split(', ') for genre in train_data['GENRE']]
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(genre_labels)

tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

with tqdm(total=50, desc="Training Model") as pbar:
    naive_bayes = MultinomialNB()
    multi_output_classifier = MultiOutputClassifier(naive_bayes)
    multi_output_classifier.fit(X_train_tfidf, y_train)
    pbar.update(50)

try:
    with tqdm(total=50, desc="Loading Test Data") as pbar:
        test_data = pd.read_csv('C://Users//yaswa//OneDrive//Desktop//vscode//CodSoft//test_data.txt', sep=':::', header=None, names=['SerialNumber', 'MOVIE_NAME', 'MOVIE_PLOT'], engine='python')
        pbar.update(50)
except Exception as e:
    print(f"Error loading test_data: {e}")

X_test = test_data['MOVIE_PLOT'].astype(str).apply(lambda doc: doc.lower())
X_test_tfidf = tfidf_vectorizer.transform(X_test)

with tqdm(total=50, desc="Predicting on Test Data") as pbar:
    y_pred = multi_output_classifier.predict(X_test_tfidf)
    pbar.update(50)

test_movie_names = test_data['MOVIE_NAME']
predicted_genres = mlb.inverse_transform(y_pred)
test_results = pd.DataFrame({'MOVIE_NAME': test_movie_names, 'PREDICTED_GENRES': predicted_genres})

test_results['PREDICTED_GENRES'] = test_results['PREDICTED_GENRES'].apply(lambda genres: [fallback_genre] if len(genres) == 0 else genres)

with open("model_evaluation.txt", "w", encoding="utf-8") as output_file:
    for _, row in test_results.iterrows():
        movie_name = row['MOVIE_NAME']
        genre_str = ', '.join(row['PREDICTED_GENRES'])
        output_file.write(f"{movie_name} ::: {genre_str}\n")

y_train_pred = multi_output_classifier.predict(X_train_tfidf)

accuracy = accuracy_score(y_train, y_train_pred)
precision = precision_score(y_train, y_train_pred, average='micro')
recall = recall_score(y_train, y_train_pred, average='micro')
f1 = f1_score(y_train, y_train_pred, average='micro')

with open("model_evaluation.txt", "a", encoding="utf-8") as output_file:
    output_file.write("\n\nModel Evaluation Metrics:\n")
    output_file.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    output_file.write(f"Precision: {precision:.2f}\n")
    output_file.write(f"Recall: {recall:.2f}\n")
    output_file.write(f"F1-score: {f1:.2f}\n")
  
print("Model evaluation results and metrics have been saved to 'model_evaluation.txt'.")

genre_counts = pd.DataFrame(mlb.inverse_transform(y_train), columns=['Genres']).explode('Genres')['Genres'].value_counts()
plt.figure(figsize=(10, 6))
genre_counts.plot(kind='bar', color='skyblue')
plt.title('Distribution of Movie Genres in Training Data')
plt.xlabel('Genres')
plt.ylabel('Counts')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

predicted_genre_counts = test_results['PREDICTED_GENRES'].explode().value_counts()
plt.figure(figsize=(10, 6))
predicted_genre_counts.plot(kind='bar', color='lightgreen')
plt.title('Distribution of Predicted Movie Genres in Test Data')
plt.xlabel('Genres')
plt.ylabel('Counts')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
