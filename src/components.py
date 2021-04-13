import concurrent.futures
import itertools
import operator
import os
import pickle
from operator import itemgetter
from gensim import corpora
import requests
import wikipedia
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import nltk
import re
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess

class AnswerExtractor:
    fileLocation: str

    def __init__(self,similarity_matrix, nlp_bert,nlp_roberta,nlp_electra,nlp_albert,nlp_xlm,stop_words, lemmatizer):
        self.similarity_matrix = similarity_matrix
        self.nlp_bert = nlp_bert
        self.nlp_roberta = nlp_roberta
        self.nlp_electra = nlp_electra
        self.nlp_albert = nlp_albert
        self.nlp_xlm = nlp_xlm
        # self.nlp = nlp
        self.stop_words = stop_words
        self.lemmatizer = lemmatizer   
        self.fileLocation = "./data_set/Lease.txt"
    
    def removeStopWords(self, text):
        filteredText = []
        for word in text:
            if not word in self.stop_words:
                filteredText.append(self.lemmatizer.lemmatize(word.lower()))

        return filteredText

    def stemWords(self, text):
        stemmedText = []
        for word in text:
            stemmedText.append(self.lemmatizer.lemmatize(word))

        return stemmedText


    def caseFold(self, text):
        foldedText = []
        for word in text:
            foldedText.append(word.lower())

        return foldedText

    def removePunctuation(self, text):
        filteredText = re.sub(r'[^\w\s]', '', text)
        return filteredText
    
    # Function for returning answers
    def getAnswers(self, Question, doc,machinecomp,wordemb):
        # Pre-process the user's question. 
        question = self.removePunctuation(Question)
        questionWords = question.split()
        questionWords = self.removeStopWords(questionWords)
        questionWords = self.caseFold(questionWords)
        questionWords = self.stemWords(questionWords)


        # Dictionary to store files and the paragraphs which were sufficiently relevant
        relevantParagraphs = dict()

        documentName = self.fileLocation
        
        paragraphs = doc.split('\n')
        
        # textFile.close()
        updatedList = []

        # Loop through the paragraphs
        for paragraph in paragraphs:
            # Pre-process the paragraph string
            cleanedText = self.removePunctuation(paragraph)
            tokens = cleanedText.split()
            tokens = self.removeStopWords(tokens)
            tokens = self.caseFold(tokens)
            tokens = self.stemWords(tokens)

            allText = [tokens, questionWords]

            # Prepare a dictionary from the paragraph and question words.
            dictionary = corpora.Dictionary(allText)

            # Convert the paragraph words and question words to bag of words vectors
            tokenVec = dictionary.doc2bow(tokens)
            questionVec = dictionary.doc2bow(questionWords)

            simVal = self.similarity_matrix[wordemb].inner_product(tokenVec, questionVec, normalized=True)                

            # Update the dictionary each time a paragraph is greater than or equal the similarity threshold.
            if simVal >= 0.17:
                if documentName in relevantParagraphs:
                    updatedList = relevantParagraphs.get(documentName)
                    pair = [simVal, paragraph]
                    updatedList.append(pair)
                    newEntry = {documentName: updatedList}
                    relevantParagraphs.update(newEntry)
                else:
                    firstPara = []
                    pair = [simVal, paragraph]
                    firstPara.append(pair)
                    newEntry = {documentName: firstPara}
                    relevantParagraphs.update(newEntry)

        if len(relevantParagraphs) == 0:
            print("SORRY! NO ANSWER AVAILABLE.")
            print("Perhaps you can try asking again with a lower threshold value.")
            print("-------------------------------------------------\n")
        else:
            answers = self.getSpan(relevantParagraphs,Question,machinecomp)
            return answers

    # Function for identifying the answer span in the paragraphs.
    def getSpan(self,relevantParagraphs,Question,machinecomp):
        # Loop through files which contain relevant paragraphs.
        for key in relevantParagraphs.keys():
            documentParas = sorted(relevantParagraphs.get(key), key=itemgetter(0), reverse=True)

            answers = []
            
            for i, para in enumerate(documentParas, 1):
                if(machinecomp==0):
                    result = self.nlp_bert(question=Question, context=para[1])
                elif(machinecomp==1):
                    result = self.nlp_roberta(question=Question, context=para[1])
                elif(machinecomp==2):
                    result = self.nlp_electra(question=Question, context=para[1])
                elif(machinecomp==3):
                    result = self.nlp_albert(question=Question, context=para[1])
                elif(machinecomp==4):
                    result = self.nlp_xlm(question=Question, context=para[1]) 
                else:
                    result = self.nlp_bert(question=Question, context=para[1])
                
                result['simscore'] = round(para[0]*100,2)
                result['text'] = para[1]
                result['score'] = round(result['score']*100,2)
                answers.append(result)

        return answers    
