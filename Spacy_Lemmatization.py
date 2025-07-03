import spacy
from prettytable import PrettyTable

#LEMMATIZATION 
# Base Form Reduction: Lemmatization reduces words to their meaningful base form(eg:'running' to 'run')
# Enhances NLP Accuracy: It normalizes word variations, improving the accuracy of the NLP tasks by treating different word forms as the same

text = 'run ran running'
nlp = spacy.load('en_core_web_md')
doc = nlp(text)
print('Working on the text :',text)

table = PrettyTable(field_names=['token','lemma'])
for token in doc:
    table.add_row([token.text,token.lemma_])

print(table)


text = 'The scientists discover new species every year. Last year, they discovered an ancient artifacts. They are discovering new techniques with their recent discovery'
print('Working on the text :',text)
doc = nlp(text)
table = PrettyTable(field_names=['token','lemma'])
for token in doc:
    table.add_row([token.text,token.lemma_])

print(table)