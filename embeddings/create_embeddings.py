import io
import pickle
import numpy as np

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data

fasttext = "crawl-300d-2M-subword/crawl-300d-2M-subword.vec"
ft_model = load_vectors(fasttext)

def words_to_vec(words):
    # fasttext vector dimension is 300
    vec = np.zeros(300)
    for word in words:
        if word not in ft_model:
            continue
        else:
            vec += ft_model.get(word)
    vec /= len(words)
    return vec


# dictionary to store embeddings
# question ID as key
# words embeddings as value
q_title_embeddings = {}
q_body_embeddings = {}
q_tags_embeddings = {}

with open('../csv_files/CS_questions.csv', 'r') as f:
    for count, line in enumerate(f):
        # skip first line
        if count == 0:
            continue
        items = line.rstrip().split(',')
        qid = items[0]
        # skip empty lines
        if qid == '':
            continue
        body_text = items[1]
        title_text = items[5]
        tags_text = items[6]
        q_body_embeddings[int(qid)] = words_to_vec(body_text.split())
        q_title_embeddings[int(qid)] = words_to_vec(title_text.split())
        if tags_text != '':
            q_tags_embeddings[int(qid)] = words_to_vec(tags_text.split())

# save into files
pickle.dump(q_title_embeddings, open("q_title_embeddings.dat", "wb"))
pickle.dump(q_body_embeddings, open("q_body_embeddings.dat", "wb"))
pickle.dump(q_tags_embeddings, open("q_tags_embeddings.dat", "wb"))


a_body_embeddings = {}

with open('../csv_files/CS_answers.csv', 'r') as f:
    for count, line in enumerate(f):
        # skip first line
        if count == 0:
            continue
        items = line.rstrip().split(',')
        aid = items[0]
        # skip empty lines
        if aid == '':
            continue
        body_text = items[3]
        if body_text == '':
            continue
        a_body_embeddings[int(aid)] = words_to_vec(body_text.split())

# save into files
pickle.dump(a_body_embeddings, open("a_body_embeddings.dat", "wb"))
