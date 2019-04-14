#!/usr/bin/env python
# coding: utf-8

# In[103]:


cs_ques_dict = pickle.load(open("../clustering/cs_ques_dict.dat","rb"))
cs_cluster_centers = pickle.load(open("../clustering/cs_cluster_centers.dat","rb"))
cs_cluster_centers_indices = pickle.load(open("../clustering/cs_cluster_centers_indices.dat","rb"))


# In[104]:


num_clusters = cs_cluster_centers.shape[0]
print("Total number of clusters: ",num_clusters)


# In[147]:


#We have the cluster centers and also the label for each question
#For a given cluster, find its 10 nearby clusters, sorted according to their distances
#nearest first
#Use euclidean distance between the cluster centers for this
from sklearn.metrics.pairwise import euclidean_distances

def computeNearByClusters(cluster_centers):
    nearby_centers_dict = {}
    for i in range(len(cluster_centers)):
        #gives the distance of i-th center from all the centers
        distances = euclidean_distances(cluster_centers,[cluster_centers[i]])
        #sort these distances
        nearest_10_centers_array = distances.argsort(axis=0)[1:6]
        nearest_10_centers = []
        for elem in nearest_10_centers_array:
            nearest_10_centers.append(elem[0])
        if not i in nearby_centers_dict:
            nearby_centers_dict[i] = nearest_10_centers
    return nearby_centers_dict


# In[148]:


#Dict where key = cluster_id, value = list of ids of 10 nearest clusters
nearby_centers_dict = computeNearByClusters(cs_cluster_centers)


# In[149]:


print(nearby_centers_dict[0])


# In[150]:


#Given a cluster, find all the questions that belong to it

def findAllQuestionsInACluster(num_clusters,questions_dict):
    questions_in_a_cluster_dict = {}
    for key, value in questions_dict.items():
        if value[1] in questions_in_a_cluster_dict:
            questions_in_a_cluster_dict[value[1]].append(key)
        else:
            questions_in_a_cluster_dict[value[1]] = [key]
    return questions_in_a_cluster_dict
    


# In[151]:


#dict where key = cluster_id
#value = list of all questions present in that cluster
questions_in_a_cluster_dict = findAllQuestionsInACluster(num_clusters,cs_ques_dict)


# In[152]:


print(questions_in_a_cluster_dict[0])


# In[116]:


import math
#given a question embedding, calculate its magnitude
def computeMagnitude(embedding):
    square_sum = 0
    for i in range(len(embedding)):
        if not math.isnan(embedding[i]):
            square_sum = square_sum + (embedding[i]*embedding[i])
    return math.sqrt(square_sum)

def computeCosine(ques_1, ques_2):
    numerator = 0
    for i in range(len(ques_1)):
        if not math.isnan(ques_1[i]) and not math.isnan(ques_2[i]):
            numerator = numerator + ques_1[i]*ques_2[i]
    denominator = computeMagnitude(ques_1) * computeMagnitude(ques_2)
    return numerator/denominator


# In[153]:


#Given a question, find which cluster it belongs to
#find all nearby clusters to that cluster
#retrieve all the questions from the nearby clusters, form a list of all the questions
#Do cosine similarity to find the questions most similar to the given question
#return top 10 of them

from sklearn.metrics.pairwise import cosine_similarity

def generateQuesRecommendations(ques_id):
    total_questions = []
    #input is question_id
    #cluster_label tells which cluster the question is in
    #cs_ques_dict is a dictionary where key is ques_id and value is a tuple(embedding,label)
    cluster_label = cs_ques_dict[ques_id][1]
    #find the 10 nearby clusters to a given cluster
    nearby_clusters = nearby_centers_dict[cluster_label]
    #nearby_clusters is a list
    for cluster in nearby_clusters:
        total_questions.extend(questions_in_a_cluster_dict[cluster])
    #total_questions is a list of all questions
    #for a given question find cosine similarity with all these questions
    cosine_scores = []
    for ques in total_questions:
        cosine_scores.append((ques,cosine_similarity(cs_ques_dict[ques_id][0].reshape(1,-1),cs_ques_dict[ques][0].reshape(1,-1))[0][0]))
    cosine_scores.sort(key=lambda x: x[1],reverse=True)
    return cosine_scores[0:10]


# In[156]:


#For a given question, generate top 10 recommendations
#output is a dict, which has ques_id as key, and value as list of tuples
#where each tuple consists of ques_id and cosine similarity score

ques_recommendations = {}
for key, value in cs_ques_dict.items():
    #here key represents the question_id
    #generate recommendation for each such question
    ques_recommendations[key] = generateQuesRecommendations(key)


# In[159]:


len(ques_recommendations)


# In[160]:


pickle.dump(ques_recommendations,open("ques_recommendations.dat","wb"))


# In[ ]:




