import os

import spacy

from flask import Flask, render_template, jsonify, request
import pickle
from operator import itemgetter
from gensim import corpora
from components import QueryProcessor, DocumentRetrieval, PassageRetrieval, AnswerExtractor, AnswerExtractorConceptnet
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import nltk
import re
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from flask_ngrok import run_with_ngrok



app = Flask(__name__)
# run_with_ngrok(app)

stop_words = set(stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()
similarity_matrix = []
# Load the similarity matrix from the saved file.
# similarity_Matrix_fasttext = pickle.load(open('similarity_matrix.sav', 'rb'))
# similarity_matrix.append(similarity_Matrix_fasttext)
similarity_Matrix_conceptnet = pickle.load(open('similarity_matrix_conceptnet.sav', 'rb'))
# similarity_Matrix_glove = pickle.load(open('similarity_matrix_glove.sav', 'rb'))
# similarity_Matrix_w2v = pickle.load(open('similarity_matrix_w2v.sav', 'rb'))
# similarity_matrix.extend((similarity_Matrix_conceptnet,similarity_Matrix_glove,similarity_Matrix_w2v))
similarity_matrix.append(similarity_Matrix_conceptnet)


# nlp = []
# tokenizer_bert = AutoTokenizer.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")
# model_bert = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-large-uncased-whole-word-masking-squad2")
# nlp_bert=pipeline("question-answering", model=model_bert, tokenizer=tokenizer_bert)
# nlp.append(nlp_bert)
nlp_bert = ""
tokenizer_roberta = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model_roberta = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
nlp_roberta=pipeline("question-answering", model=model_roberta, tokenizer=tokenizer_roberta)

# tokenizer_electra = AutoTokenizer.from_pretrained("deepset/electra-base-squad2")
# model_electra = AutoModelForQuestionAnswering.from_pretrained("deepset/electra-base-squad2")
# nlp_electra =pipeline("question-answering", model=model_electra, tokenizer=tokenizer_electra)
nlp_electra = ""
# tokenizer_albert = AutoTokenizer.from_pretrained("mfeb/albert-xxlarge-v2-squad2")
# model_albert = AutoModelForQuestionAnswering.from_pretrained("mfeb/albert-xxlarge-v2-squad2")
# nlp_albert=pipeline("question-answering", model=model_albert, tokenizer=tokenizer_albert)
nlp_albert = ""
# tokenizer_xlm = AutoTokenizer.from_pretrained("deepset/xlm-roberta-large-squad2")
# model_xlm = AutoModelForQuestionAnswering.from_pretrained("deepset/xlm-roberta-large-squad2")
# nlp_xlm =pipeline("question-answering", model=model_xlm, tokenizer=tokenizer_xlm)
nlp_xlm = ""
# nlp.extend((nlp_roberta,tokenizer_electra,tokenizer_albert,tokenizer_xlm))

# answer_extractor = AnswerExtractorConceptnet(similarity_Matrix_fasttext,similarity_Matrix_conceptnet,similarity_Matrix_glove,similarity_Matrix_w2v,nlp_bert, nlp_roberta,nlp_electra,nlp_albert,nlp_xlm,stop_words,lemmatizer)
answer_extractor = AnswerExtractorConceptnet(similarity_matrix,nlp_bert,nlp_roberta,nlp_electra,nlp_albert,nlp_xlm,stop_words,lemmatizer)



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/answer-question', methods=['POST'])
def analyzer():
    data = request.get_json()
    print(data.get('datanum'))
    question = data.get('question')
    machinecomp = data.get('machinecomp')
    print(machinecomp)
    wordemb = data.get('wordemb')
    print(wordemb)
    document = data.get('doc')
    answers = answer_extractor.getAnswers(question, document,machinecomp,wordemb)
    print(answers)
    return jsonify(answers)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=5000)
    app.run()
