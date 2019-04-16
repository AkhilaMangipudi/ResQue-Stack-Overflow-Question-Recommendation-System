import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import tqdm as tqdm
from collections import defaultdict
import pickle as pickle

def get_tfidf(column):
	tfidf_vectorizer = TfidfVectorizer(stop_words ="english")
	tfidf_matrix = tfidf_vectorizer.fit_transform(column.values.astype(str))
	return tfidf_matrix

def find_similar(tfidf_matrix, index, top_n = 5):
    cosine_similarities = linear_kernel(tfidf_matrix[index:index+1], tfidf_matrix).flatten()
    related_docs_indices = [i for i in cosine_similarities.argsort()[::-1] if i != index]
    return [(index, cosine_similarities[index]) for index in related_docs_indices][0:top_n]

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)	

answers_file = "CS_answers.csv"
df = pd.read_csv(answers_file)
saved_column = df.Body #you can also use df['column_name']
saved_column.fillna(" ")
tfidf_answers = get_tfidf(saved_column)

answers_rec_dict = defaultdict(list)
for i in tqdm.tqdm(range(saved_column.shape[0])):
    for index, score in find_similar(tfidf_answers, i):
        if df['Id'][i] in answers_rec_dict:
            (answers_rec_dict[df['Id'][i]]).append(saved_column[index])
        else:
            answers_rec_dict[df['Id'][i]] = [saved_column[index]]

save_obj(questions_rec_dict,"tf_idf_answers_rec")
			

questions_file = "CS_questions.csv"
df_ques = pd.read_csv(questions_file)
df_question_merged = df_ques.Body.astype(str) + " " + df_ques.Title.astype(str) + " " + df_ques.Tags.astype(str)
tfidf_questions_merged = get_tfidf(df_question_merged)

questions_rec_dict = defaultdict(list)
for i in tqdm.tqdm(range(tfidf_questions_merged.shape[0])):
    for index, score in find_similar(tfidf_questions_merged, i):
        if df_ques['Id'][i] in questions_rec_dict:
            (questions_rec_dict[df_ques['Id'][i]]).append(df_question_merged[index])
        else:
            questions_rec_dict[df_ques['Id'][i]] = [df_question_merged[index]]

save_obj(questions_rec_dict,"tf_idf_ques_rec")
