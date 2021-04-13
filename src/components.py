import concurrent.futures
import itertools
import operator
import os
import pickle
from operator import itemgetter
from gensim import corpora
import requests
import wikipedia
from gensim.summarization.bm25 import BM25
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import nltk
import re
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess


class QueryProcessor:

    def __init__(self, nlp, keep=None):
        self.nlp = nlp
        self.keep = keep or {'PROPN', 'NUM', 'VERB', 'NOUN', 'ADJ'}

    def generate_query(self, text):
        doc = self.nlp(text)
        query = ' '.join(token.text for token in doc if token.pos_ in self.keep)
        return query


class DocumentRetrieval:

    def __init__(self, url='https://en.wikipedia.org/w/api.php'):
        self.url = url

    def search_pages(self, query):
        params = {
            'action': 'query',
            'list': 'search',
            'srsearch': query,
            'format': 'json'
        }
        res = requests.get(self.url, params=params)
        return res.json()

    def search_page(self, page_id):
        res = wikipedia.page(pageid=page_id)
        return res.content

    def search(self, query):
        pages = self.search_pages(query)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            process_list = [executor.submit(self.search_page, page['pageid']) for page in pages['query']['search']]
            docs = [self.post_process(p.result()) for p in process_list]
        return docs

    def post_process(self, doc):
        pattern = '|'.join([
            '== References ==',
            '== Further reading ==',
            '== External links',
            '== See also ==',
            '== Sources ==',
            '== Notes ==',
            '== Further references ==',
            '== Footnotes ==',
            '=== Notes ===',
            '=== Sources ===',
            '=== Citations ===',
        ])
        p = re.compile(pattern)
        indices = [m.start() for m in p.finditer(doc)]
        min_idx = min(*indices, len(doc))
        return doc[:min_idx]


class PassageRetrieval:

    def __init__(self, nlp):
        self.tokenize = lambda text: [token.lemma_ for token in nlp(text)]
        self.bm25 = None
        self.passages = None

    def preprocess(self, doc):
        passages = [p for p in doc.split('\n') if p and not p.startswith('=')]
        return passages

    def fit(self, docs):
        passages = list(itertools.chain(*map(self.preprocess, docs)))
        corpus = [self.tokenize(p) for p in passages]
        self.bm25 = BM25(corpus)
        self.passages = passages

    def most_similar(self, question, topn=10):
        tokens = self.tokenize(question)
        scores = self.bm25.get_scores(tokens)
        pairs = [(s, i) for i, s in enumerate(scores)]
        pairs.sort(reverse=True)
        passages = [self.passages[i] for _, i in pairs[:topn]]
        return passages


class AnswerExtractor:

    def __init__(self, tokenizer, model):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        model = AutoModelForQuestionAnswering.from_pretrained(model)
        self.nlp = QuestionAnsweringPipeline(model=model, tokenizer=tokenizer)

    def extract(self, question, passages):
        answers = []
        for passage in passages:
            try:
                answer = self.nlp(question=question, context=passage)
                answer['text'] = passage
                answers.append(answer)
            except KeyError:
                pass
        answers.sort(key=operator.itemgetter('score'), reverse=True)
        return answers

class AnswerExtractorConceptnet:
    fileLocation: str

    def __init__(self,similarity_matrix, nlp_bert,nlp_roberta,nlp_electra,nlp_albert,nlp_xlm,stop_words, lemmatizer):
        # self.similarity_matrix_fasttext = similarity_Matrix_fasttext
        # self.similarity_matrix_conceptnet = similarity_Matrix_conceptnet
        # self.similarity_matrix_glove = similarity_Matrix_glove
        # self.similarity_matrix_w2v = similarity_Matrix_w2v

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
    
    # Function for removing stop words from a given list of words.
    # It also lemmatizes and case folds the words to ease comparisons.
    def removeStopWords(self, text):
        filteredText = []
        for word in text:
            if not word in self.stop_words:
                filteredText.append(self.lemmatizer.lemmatize(word.lower()))

        return filteredText

    # Function for lemmatising tokens
    def stemWords(self, text):
        stemmedText = []
        for word in text:
            stemmedText.append(self.lemmatizer.lemmatize(word))

        return stemmedText


    # Function for case folding tokens
    def caseFold(self, text):
        foldedText = []
        for word in text:
            foldedText.append(word.lower())

        return foldedText

    # Function for removing punctuation from a string.
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
        # # Ensure that the file exists before proceeding.
        # textFile = open(self.fileLocation, "r",encoding='utf-8')
        # paras = doc.split('\n\n')
        # print(paras)
        # Split document text into separate paragraphs
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
            # if(wordemb==0):
            #     # Using inner product with normalisation has same effect as soft cosine similarity.
            #     simVal = self.similarity_matrix_fasttext.inner_product(tokenVec, questionVec, normalized=True)
            # elif(wordemb==1):
            #     # Using inner product with normalisation has same effect as soft cosine similarity.
            #     simVal = self.similarity_matrix_conceptnet.inner_product(tokenVec, questionVec, normalized=True)
            # elif(wordemb==2):
            #     # Using inner product with normalisation has same effect as soft cosine similarity.
            #     simVal = self.similarity_matrix_glove.inner_product(tokenVec, questionVec, normalized=True)
            # elif(wordemb==3):
            #     # Using inner product with normalisation has same effect as soft cosine similarity.
            #     simVal = self.similarity_matrix_w2v.inner_product(tokenVec, questionVec, normalized=True) 
            # else:
            #     simVal = self.similarity_matrix_fasttext.inner_product(tokenVec, questionVec, normalized=True)                

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
#           joinedText = '\n'.join(documentParas) # Put relevant paragraphs into one string to get better answer span.

            answers = []
            # Print out each paragraph with the answer span highlighted.
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
                # nlp = self.nlp[machinecomp]
                # result = nlp(question=Question, context=para[1])
                result['simscore'] = round(para[0]*100,2)
                result['text'] = para[1]
                result['score'] = round(result['score']*100,2)
                answers.append(result)

        return answers    
