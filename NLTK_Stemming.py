import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
import spacy
from nltk.stem import PorterStemmer
from prettytable import PrettyTable

nlp = spacy.load('en_core_web_lg')
spacy_stop_words = nlp.Defaults.stop_words


## Stemming 
# Stemming reduces word into base form 

## Similar to Lemmatization
## The words will be converted into base in both stemming and lemmatization but lemmatization will convert into a base word form which makes sense 
## In case of stemming the base word will not make much sense

text = 'The scientists discover new species every year. Last year, they discovered an ancient artifacts. They are discovering new techniques with their recent discovery'

stemmer = PorterStemmer()

words = word_tokenize(text)

stemmed_words = [stemmer.stem(word) for word in words]

table = PrettyTable()
table.field_names=['Stem words','Tokenized words']

for i,j in zip(stemmed_words,words):
    table.add_row([i,j])

print(table)