#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Parse the Posts.xml file to get Question and answer formation
import xml.etree.ElementTree as ET
tree = ET.parse('../IR_datasets/cs.stackexchange.com/Posts.xml')
root = tree.getroot()


# In[2]:


root.tag


# In[3]:


#Pre-process the sentence by removing all the HTML tags and stopwords
from bs4 import BeautifulSoup
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

#this translator is used to remove the punctuation
translator = str.maketrans('', '', string.punctuation)

def sentencePreProcess(body):
    #this removes all the HTML tags
    clean_body = BeautifulSoup(body,"lxml").text
    #this removes all the punctuation
    clean_body = clean_body.translate(translator)
    #tokenize the given sentence
    word_tokens = word_tokenize(clean_body)
    #remove the stop words
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    #convert from list to sentence
    body = ' '.join(word for word in filtered_sentence)
    return body


# In[5]:


import csv

#creates the CSV files for questions and answers
#root: root of the XML tree
#quesFileName: question csv file name along with the path
#ansFileName: answer-csv file name along with the path
def createCSV(root,quesFileName,ansFileName):
    with open(quesFileName,mode='w',encoding='utf-8') as ques_file, open(ansFileName,mode='w',encoding='utf-8') as ans_file:
        ques_file = csv.writer(ques_file, delimiter=',')
        ans_file = csv.writer(ans_file, delimiter=',')
        ques_file.writerow(['Id','Body','AcceptedAnswerId','Score','ViewCount','Title','Tags','AnswerCount','CommentCount','FavoriteCount'])
        ans_file.writerow(['Id', 'ParentId','Score','Body','CommentCount'])
        for row in root.findall('row'):
            post_type_id = row.get("PostTypeId")
            if(post_type_id == "1"):
                #It is a question
                post_id = row.get('Id')
                body = sentencePreProcess(row.get('Body'))
                accepted_answer_id = row.get('AcceptedAnswerId')
                score = row.get('Score')
                view_count = row.get('ViewCount')
                title = sentencePreProcess(row.get('Title'))
                tags = sentencePreProcess(row.get('Tags'))
                answer_count = row.get('AnswerCount')
                comment_count = row.get('CommentCount')
                fav_count = row.get('FavoriteCount')
                ques_file.writerow([post_id,body,accepted_answer_id,score,view_count,title,tags,answer_count,comment_count,fav_count])
            elif(post_type_id == "2"):
                #it is an answer
                post_id = row.get('Id')
                parent_id = row.get('ParentId')
                score = row.get('Score')
                body = sentencePreProcess(row.get('Body'))
                comment_count = row.get('CommentCount')
                ans_file.writerow([post_id,parent_id,score,body,comment_count])
    
    
    


# In[6]:


createCSV(root,"../csv_files/CS_questions.csv","../csv_files/CS_answers.csv")

