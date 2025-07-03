import nltk
from nltk.tokenize import word_tokenize,sent_tokenize
from prettytable import PrettyTable

# nltk.download('punkt')
text = '"Lord of the Rings" refers to a high fantasy novel by J.R.R. Tolkien, published in three volumes between 1954 and 1955, and later adapted into a popular film trilogy directed by Peter Jackson. The story, set in the fictional world of Middle-earth, follows the hobbit Frodo Baggins and his companions on a quest to destroy the One Ring, which holds the power of the Dark Lord Sauron. '

words = word_tokenize(text)
sentences = sent_tokenize(text)

print('Splitting the text into words using word_tokenize----> done')
print(words)
print('Splitting the text into sentences using sent_tokenize----> done')
print(sentences)


##### 2. Filtering Stop Words
# Stop words are common words that are usually removed in text preprocessing task

from nltk.corpus import stopwords
import spacy

#Before using stopwords from nltk we have to download the stopwords
# nltk.download('stopwords')

stop_words = stopwords.words('english')
print("Downloaded the stop words")
print(stop_words)
print('Length of NLP stop words without spacy is :',len(stop_words))#198

#Let us use spacy model 
nlp = spacy.load('en_core_web_lg')
spacy_stop_words = nlp.Defaults.stop_words
print('Created stop words from spacy large model ')
print(spacy_stop_words)
print('Length of the stop words in spacy large model is :',len(spacy_stop_words)) # 326

text = " ".join(open('bert_extracted_text.txt').read().splitlines())#WHole text data from the file
words = word_tokenize(text.lower())

#Let us get the count of stop words in words
print(len(set([word for word in words if word in spacy_stop_words])))
