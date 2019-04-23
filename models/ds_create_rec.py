import pickle
import numpy as np

ds_ques_dict = pickle.load(open("ds_title_ques_dict.dat","rb"))
ds_cluster_centers = pickle.load(open("ds_title_cluster_centers.dat","rb"))
#ds_cluster_centers_indices = pickle.load(open("ds_cluster_centers_indices.dat","rb"))

num_clusters = ds_cluster_centers.shape
print("cluster_centers shape:", num_clusters)

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

nearby_centers_dict = computeNearByClusters(ds_cluster_centers)
print(nearby_centers_dict[2])

def findAllQuestionsInACluster(questions_dict):
    questions_in_a_cluster_dict = {}
    for key, value in questions_dict.items():
        if value[1] in questions_in_a_cluster_dict:
            questions_in_a_cluster_dict[value[1]].append(key)
        else:
            questions_in_a_cluster_dict[value[1]] = [key]
    return questions_in_a_cluster_dict

questions_in_a_cluster_dict = findAllQuestionsInACluster(ds_ques_dict)

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

from sklearn.metrics.pairwise import cosine_similarity

def generateQuesRecommendations(ques_id):
    total_questions = []
    #input is question_id
    #cluster_label tells which cluster the question is in
    #ds_ques_dict is a dictionary where key is ques_id and value is a tuple(embedding,label)
    cluster_label = ds_ques_dict[ques_id][1]
    #find the 10 nearby clusters to a given cluster
    nearby_clusters = nearby_centers_dict[cluster_label]
    #nearby_clusters is a list
    for cluster in nearby_clusters:
        total_questions.extend(questions_in_a_cluster_dict[cluster])
    total_questions.extend(questions_in_a_cluster_dict[cluster_label])
    #total_questions is a list of all questions
    #for a given question find cosine similarity with all these questions
    cosine_scores = []
    for ques in total_questions:
        cosine_scores.append((ques,cosine_similarity(ds_ques_dict[ques_id][0].reshape(1,-1),ds_ques_dict[ques][0].reshape(1,-1))[0][0]))
    cosine_scores.sort(key=lambda x: x[1],reverse=True)
    return cosine_scores[0:21]

ds_ques_recommendations = {}
for key, value in ds_ques_dict.items():
    #here key represents the question_id
    #generate recommendation for each such question
    ds_ques_recommendations[key] = generateQuesRecommendations(key)

pickle.dump(ds_ques_recommendations,open("ds_title_ques_recommendations.dat","wb"))
