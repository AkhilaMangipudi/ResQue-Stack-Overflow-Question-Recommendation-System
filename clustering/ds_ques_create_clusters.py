import pickle
import numpy as np
from sklearn.cluster import AffinityPropagation

ds_q_title_embeddings = pickle.load(open("./ds_embeddings/ds_q_title_embeddings.dat","rb"))
ds_q_body_embeddings = pickle.load(open("./ds_embeddings/ds_q_body_embeddings.dat","rb"))
ds_q_tags_embeddings = pickle.load(open("./ds_embeddings/ds_q_tags_embeddings.dat","rb"))

def makeQuesDict(q_title_embeddings,q_body_embeddings,q_tags_embeddings):
    ds_ques_dict = {}
    for key, value in q_body_embeddings.items():
        body_embedding = value
        title_embedding = np.zeros(300)
        tag_embedding = np.zeros(300)
        if key in q_title_embeddings:
        #this question has a title
            title_embedding = q_title_embeddings[key]
        if key in q_tags_embeddings:
        #this question has a tag
            tag_embedding = q_tags_embeddings[key]
        final_embedding = body_embedding + title_embedding + tag_embedding
        ds_ques_dict[key] = final_embedding
    return ds_ques_dict

ds_ques_dict = makeQuesDict(ds_q_title_embeddings,ds_q_body_embeddings,ds_q_tags_embeddings)

def makeQuesEmbeddingsArray(questions_dict):
    ques_embeddings_array = np.zeros((len(questions_dict),300))
    ques_id_list = []
    i=0
    for key, value in questions_dict.items():
        ques_embeddings_array[i,:] = value
        ques_id_list.append(key)
        i = i+1
    return ques_embeddings_array, ques_id_list

ds_ques_embeddings_array, ds_ques_id_list = makeQuesEmbeddingsArray(ds_ques_dict)

clustering = AffinityPropagation().fit(ds_ques_embeddings_array)

#Append the labels of each question to the ques dict
for key, value in ds_ques_dict.items():
    #get the index from the ques_id_list
    idx = ds_ques_id_list.index(key)
    new_value = (value, clustering.labels_[idx])
    ds_ques_dict[key] = new_value

#save the cluster centers
pickle.dump(clustering.cluster_centers_,open("ds_cluster_centers.dat","wb"))
pickle.dump(clustering.cluster_centers_indices_,open("ds_cluster_centers_indices.dat","wb"))
pickle.dump(clustering.labels_,open("ds_cluster_labels.dat","wb"))

#save the question_list
pickle.dump(ds_ques_dict,open("ds_ques_dict.dat","wb"))
pickle.dump(ds_ques_embeddings_array, open("ds_ques_embeddings_array.dat", "wb"))
pickle.dump(ds_ques_id_list,open("ds_ques_id_list.dat","wb"))
