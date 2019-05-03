# ResQue-Stack-Overflow-Question-Recommendation-System
Stack Overflow Questions Recommendation System

Steps followed to generate Question Recommendations:

## Pre-processing:
The Posts.xml file which has all the question and answer information is parsed into Questions.csv and Answers.csv
run pre-processing/data_preprocessing.py and csv files will be generated in csv_files directory

## Create embeddings:
The question body, title and tag information is taken from the csv files and word embeddings are created for each of them
run embeddings/create_embeddings.py and the question_title, question_body, question_tag, answer_body embeddings are created in embeddings folder.

## Clustering:
For feature vector representation, combinations of only title, title+body and title+body+tags were experimented upon.
The final word embedding for a question is formed based on the combination we wish to use.
Clustering is applied on all the questions and the technique used is Affinity Propagation
run clustering/clustering.py and the cluster centers and the cluster labels are stored in clustering directory

## Question Recommendations:
1. Given a question, find which cluster the question belongs to
2. Find the 5 nearby clusters to the given cluster
3. Assemble all the questions from the nearby clusters and the given cluster
4. Compute cosine similarity and rank the questions according to their similarity to the given question

Additionally, answer information was also used to generate question recommendations. The hypothesis used was: If two answers are similar, then their questions are also similar. The first 5 recommendations from question and answer information were interleaved to form the final ranked list

run models/questions_embeddings_cosine.py and it generates ques_recommendations.dat which has information of each question 
and 10 most similar questions to the given question.

## Hybrid Recommendations:
To capture the rare terms as well as the semantic relations, combination of tf-idf and word embeddings were used. 
run models/hybrid_recommendations.py to generate hybrid recommendations for a given question

## Evaluation:
1. Read PostLinks.xml and club related question ids and answer ids for a particular question.
2. Evaluate our model by taking weighted average of the number of questions from relatedID list (from above step) that's present in the list of recommendations and the number of questions from relatedID list that's present in the top half of the list of recommendations



