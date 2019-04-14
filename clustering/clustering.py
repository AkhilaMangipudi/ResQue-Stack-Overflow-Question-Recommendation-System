#!/usr/bin/env python
# coding: utf-8

# In[21]:


#Load the dat files for embeddings
#Load the dat files from pickle
import pickle
cs_q_title_embeddings = pickle.load(open("../embeddings/cs_q_title_embeddings.dat","rb"))
cs_q_body_embeddings = pickle.load(open("../embeddings/cs_q_body_embeddings.dat","rb"))
cs_q_tags_embeddings = pickle.load(open("../embeddings/cs_q_tags_embeddings.dat","rb"))


# In[22]:


#ques_dict is a dictionary with key as ques_id and value as ques_embedding
#Each question embedding is formed by adding its title, tag and body embeddings
#after clustering, label will also be added as a tuple along with the embedding.

import numpy as np
def makeQuesDict(q_title_embeddings,q_body_embeddings,q_tags_embeddings):
    cs_ques_dict = {}
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
        cs_ques_dict[key] = final_embedding
    return cs_ques_dict


# In[23]:


cs_ques_dict = makeQuesDict(cs_q_title_embeddings,cs_q_body_embeddings,cs_q_tags_embeddings)


# In[24]:


print("Length of ques dict is: ",len(cs_ques_dict))
#Sample question in the question_list
print(cs_ques_dict[2])


# In[25]:


#Clustering
#For clustering, we only need the question embeddings
#Make an array of just the question_embeddings to give 
#as input to clustering algorithm

#this function returns
#ques_embeddings_array: an array of question embeddings
#ques_id_list: list of ques_id which corresponds to each embedding

def makeQuesEmbeddingsArray(questions_dict):
    ques_embeddings_array = np.zeros((len(questions_dict),300))
    ques_id_list = []
    i=0
    for key, value in questions_dict.items():
        ques_embeddings_array[i,:] = value
        ques_id_list.append(key)
        i = i+1
    return ques_embeddings_array, ques_id_list


# In[26]:


cs_ques_embeddings_array, cs_ques_id_list = makeQuesEmbeddingsArray(cs_ques_dict)

#This goes as input to the clustering algorithm


# In[27]:


print("Input to clustering algorithm shape: ",cs_ques_embeddings_array.shape)


# In[ ]:


#Clustering
from sklearn.cluster import AffinityPropagation
clustering = AffinityPropagation().fit(cs_ques_embeddings_array)


# In[ ]:


#Append the labels of each question to the ques dict
for key, value in cs_ques_dict.items():
    #get the index from the ques_id_list
    idx = ques_id_list.index(key)
    new_value = (value,clustering.labels_[idx])
    cs_ques_dict[key] = new_value


# In[ ]:


#save the cluster centers
pickle.dump(clustering.cluster_centers_,open("cs_cluster_centers.dat","wb"))
pickle.dump(clustering.cluster_centers_indices_,open("cs_cluster_centers_indices.dat","wb"))
pickle.dump(clustering.labels_,open("cs_cluster_labels.dat","wb"))

#save the question_list
pickle.dump(cs_ques_dict,open("cs_ques_dict.dat","wb"))
pickle.dump(cs_ques_embeddings_array, open("cs_ques_embeddings_array.dat", "wb"))
pickle.dump(cs_ques_id_list,open("cs_ques_id_list","wb"))

