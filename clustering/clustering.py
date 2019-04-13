#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Load the dat files for embeddings
#Load the dat files from pickle
import pickle
cs_q_title_embeddings = pickle.load(open("../embeddings/cs_q_title_embeddings.dat","rb"))
cs_q_body_embeddings = pickle.load(open("../embeddings/cs_q_body_embeddings.dat","rb"))
cs_q_tags_embeddings = pickle.load(open("../embeddings/cs_q_tags_embeddings.dat","rb"))


# In[3]:


#question_list is a list of dictionaries
#Each dict represents a Question with
# 'id': question_id, 'embedding': ques_embedding
#Each question embedding is formed by adding its title, tag and body embeddings
#after clustering, label will also be added

import numpy as np
def makeQuesList(q_title_embeddings,q_body_embeddings,q_tags_embeddings):
    question_list = []
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
        question_list.append({'id':key, 'embedding': final_embedding})
    return question_list


# In[4]:


cs_question_list = makeQuesList(cs_q_title_embeddings,cs_q_body_embeddings,cs_q_tags_embeddings)


# In[5]:


#Sample question in the question_list
print(cs_question_list[0])


# In[6]:


#Clustering
#For clustering, we only need the question embeddings
#Make an array of just the question_embeddings to give 
#as input to clustering algorithm

def makeQuesEmbeddingsArray(question_list):
    ques_embeddings = np.zeros((len(question_list),300))
    for i in range(len(question_list)):
        ques_embeddings[i,:] = question_list[i]['embedding']
    return ques_embeddings


# In[7]:


cs_ques_embeddings = makeQuesEmbeddingsArray(cs_question_list)

#This goes as input to the clustering algorithm


# In[8]:


print("Input to clustering algorithm shape: ",cs_ques_embeddings.shape)


# In[ ]:


#Clustering
from sklearn.cluster import AffinityPropagation
clustering = AffinityPropagation().fit(cs_ques_embeddings)


# In[ ]:


#Append the labels of each question to the ques dict
for i in range(len(cs_question_list)):
        cs_question_list[i]['label'] = clustering.labels_[i]


# In[ ]:


#save the cluster centers
pickle.dump(clustering.cluster_centers_,open("cs_cluster_centers.dat","wb"))
pickle.dump(clustering.cluster_centers_indices_,open("cs_cluster_centers_indices.dat","wb"))

#save the question_list
pickle.dump(cs_question_list,open("cs_question_list.dat","wb"))

