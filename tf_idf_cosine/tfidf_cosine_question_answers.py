import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def get_tfidf(column):
	tfidf_vectorizer = TfidfVectorizer(stop_words ="english")
	tfidf_matrix = tfidf_vectorizer.fit_transform(column.values.astype(str))
	return tfidf_matrix

def find_similar(tfidf_matrix, index, top_n = 5):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]
	

answers_file = "CS_answers.csv"
df = pd.read_csv(answers_file)
saved_column = df.Body #you can also use df['column_name']
saved_column.fillna(" ")
tfidf_answers = get_tfidf(saved_column)
for index, score in find_similar(tfidf_answers, 10):
    print(score, saved_column[index])
	

questions_file = "CS_questions.csv"
df_ques = pd.read_csv(questions_file)
df_question_merged = df_ques.Body.astype(str) + " " + df_ques.Title.astype(str) + " " + df_ques.Tags.astype(str)
tfidf_questions_merged = get_tfidf(df_question_merged)
for index, score in find_similar(tfidf_questions_merged, 0):
    print(score, df_question_merged[index])
