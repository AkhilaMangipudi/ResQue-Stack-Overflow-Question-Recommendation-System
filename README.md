# ResQue-Stack-Overflow-Question-Recommendation-System
Stack Overflow Questions Recommendation System

Steps followed to generate Question Recommendations:

## Pre-processing:
The Posts.xml file which has all the question and answer information is parsed into Questions.csv and Answers.csv
run pre-processing/data_preprocessing.py and csv files will be generated in csv_files directory

## Create embeddings:
The question body, title and tag information is taken from the csv files and word embeddings are created for each of them
The final word embedding for a question is formed by adding up the body,title and tag embeddings
run embeddings/create_embeddings.py and embeddings will be created in embeddings folder

## Clustering:
Clustering is applied on all the questions and the technique used is Affinity Propagation
run clustering/clustering.py and the cluster centers and the cluster labels are stored in clustering directory

## Question Recommendations:
1. Given a question, find which cluster the question belongs to
2. Find the 5 nearby clusters to the given cluster
3. Assemble all the questions from the nearby clusters and the given cluster
4. Compute cosine similarity and rank the questions according to their similarity to the given question

run models/questions_embeddings_cosine.py and it generates ques_recommendations.dat which has information of each question 
and 10 most similar questions to the given question.

## Evaluation:
1. Read PostLinks.xml and club related question ids and answer ids for a particular question.
2. Evaluate our model by taking weighted average of the number of questions from relatedID list (from above step) that's present in the list of recommendations and the number of questions from relatedID list that's present in the top half of the list of recommendations



