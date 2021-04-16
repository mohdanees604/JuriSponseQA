import nltk
import re
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess



stop_words = set(stopwords.words('english'))
lemmatizer = nltk.stem.WordNetLemmatizer()

# Function for getting text from all files in the given list.
def ReadTxtFiles(files):
    for fname in files:
        for line in open(fname):
            yield simple_preprocess(line)


# Function for removing stop words
def removeStopWords(text):
    stop_words = set(stopwords.words('english'))
    filteredText = []
    for word in text:
        if not word in stop_words:
            filteredText.append(word)

    return filteredText


# Function for lemmatising tokens
def stemWords(text):
    stemmedText = []
    for word in text:
        stemmedText.append(lemmatizer.lemmatize(word))

    return stemmedText


# Function for case folding tokens
def caseFold(text):
    foldedText = []
    for word in text:
        foldedText.append(word.lower())

    return foldedText


# Function for removing punctuation from a string.
def removePunctuation(text):
    filteredText = re.sub(r'[^\w\s]', '', text)
    return filteredText

