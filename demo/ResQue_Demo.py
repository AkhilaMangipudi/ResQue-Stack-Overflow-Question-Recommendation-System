
import tkinter
from tkinter import *
import io
import pickle
import numpy as np
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import preprocessing

root = Tk()

# set the background colour of GUI window
root.configure(background='light gray')

# set the title of GUI window
root.title("ResQue Evaluation")

# set the configuration of GUI window
root.geometry("900x800")

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return data

fasttext = "../crawl-300d-2M-subword/crawl-300d-2M-subword.vec"
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

def recommend_temp():
    hybrid_recommendations = pickle.load(open("../models/ds_que_ans_withtags_recommendations.dat","rb"))
    ques_id = 20
    input_df = pd.read_csv("../csv_files/posts_body.csv")
    recommended_questions = []
    for tup in hybrid_recommendations[ques_id]:
        recommended_questions.append(input_df.loc[input_df['Id'] == tup[0], 'Title'].iloc[0])
    recommended.delete(0.0, tkinter.END)
    for i in range (0,10):
        recommended.insert(END, recommended_questions[i] + '\n')
        recommended.update()
    return recommended_questions

# Driver code
def recommend():
    recommended_questions = []
    print("in recommend()")
    
    title_embedding = words_to_vec((ques_title.get('1.0', END)).split())
    title_embedding = preprocessing.normalize([title_embedding],norm='l2')[0]
    body_embedding = words_to_vec((ques_body.get('1.0', END)).split())
    body_embedding = preprocessing.normalize([body_embedding],norm='l2')[0]
    tag_embedding = words_to_vec((ques_tag.get('1.0', END)).split())
    tag_embedding = preprocessing.normalize([tag_embedding],norm='l2')[0]

    ques_embedding = title_embedding + body_embedding + tag_embedding
    #Step 2: Find the cluster to which this question belongs to
    #based on the distance of question embeddings from the cluster centers

    cluster_centers = pickle.load(open("../clustering/n_ds_withtags_cluster_centers.dat","rb"))
    

    distances = euclidean_distances(cluster_centers,[ques_embedding])
    given_ques_cluster = distances.argsort(axis=0)[0][0]

    
    nearby_centers_dict = pickle.load(open("../models/ds_nearby_centers_dict.dat","rb"))
    questions_in_a_cluster_dict = pickle.load(open("../models/n_ds_withtags_questions_in_a_cluster_dict.dat","rb"))
    bio_ques_dict = pickle.load(open("../clustering/n_ds_withtags_ques_dict.dat","rb"))
    input_df = pd.read_csv("../csv_files/posts_body.csv")

    total_questions = []
    
    cluster_label = given_ques_cluster
    nearby_clusters = nearby_centers_dict[cluster_label]
    #nearby_clusters is a list
    for cluster in nearby_clusters:
        total_questions.extend(questions_in_a_cluster_dict[cluster])
        #total_questions is a list of all questions
        #for a given question find cosine similarity with all these questions
        cosine_scores = []
        for ques in total_questions:
            cosine_embedding = cosine_similarity(ques_embedding.reshape(1,-1),bio_ques_dict[ques][0].reshape(1,-1))
            cosine_scores.append((ques,(cosine_embedding)[0][0]))
        cosine_scores.sort(key=lambda x: x[1],reverse=True)
        for tup in cosine_scores:
            #get the ques_id and its title from the dataframe
            #print(tup[0])
            recommended_questions.append(input_df.loc[input_df['Id'] == tup[0], 'Title'].iloc[0])
    recommended.delete(0.0, tkinter.END)
    for i in range (0,10):
        recommended.insert(END, recommended_questions[i] + '\n')
        recommended.update()
    print(recommended_questions[0:100])
    return recommended_questions[0:10]


ques_body = Text(root, height = 5, width = 20)
ques_title = Text(root, height = 5, width = 20)
ques_tag = Text(root, height = 5, width = 20)

recommended = Text(root, height = 15, width = 20)

def clear():
    # clear the content of text entry box
    name_field.delete(0, END)
    course_field.delete(0, END)


if __name__ == "__main__":
    
    heading = Label(root, text="Form", bg="light gray")

    # create a Name label
    name = Label(root, text="Enter Question Body", bg="light gray")

    # create a Course label
    course = Label(root, text="Enter Question Title", bg="light gray")
    
    tag = Label(root, text="Enter Question Tags", bg="light gray")

    rec_list = Label(root, text ="Recommended List :", bg = "light gray")
    dummy = Label(root, text =" ", bg = "light gray")
    dummy1 = Label(root, text =" ", bg = "light gray")
    dummy2 = Label(root, text=" ", bg = "light gray")
    heading.grid(row=0, column=1)
    
    course.grid(row=1, column=0)
    dummy.grid(row=2,column=0)
    name.grid(row=3, column=0)
    dummy1.grid(row=4,column=0)
    tag.grid(row=5,column=0)
    dummy2.grid(row=6, column=0)
    rec_list.grid(row=7, column=0)
    

    ques_body.grid(row=1, column=1, ipadx="100")
    ques_title.grid(row=3, column=1, ipadx="100")
    ques_tag.grid(row=5, column=1, ipadx="100")
    recommended.grid(row=7, column=1, ipadx="200")

    submit = Button(root, text="Recommend", fg="Black", bg="Gray",command=recommend)
    submit.grid(row=9, column=1)
    #recommend = recommended_questions
    root.mainloop()
