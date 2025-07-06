import spacy
from prettytable import PrettyTable
nlp = spacy.load('en_core_web_md')

text = "Hello Bharghav, try to be good and learn spacy.You are 26 years old. You can't be that lazy"
doc = nlp(text)

print('Let us see what the doc variable has in the spacy\n',doc)
print('Type of doc :',type(doc)) # <class 'spacy.tokens.doc.Doc'>

#Let us see what do we have in this spacy doc
# print('Let us get everything in the spacy doc \n',doc.to_dict()) # This is not good to have this shown 

#TOKENIZATION 
print('Checking whether the token is character or not and using ".is_alpha" field')
for token in doc:
    print(token,'--->',token.is_alpha)
    #.is_alpha ----> Checks whether the token is characters or not(True or false)

print('Checking whether the token is punct or not')
for token in doc:
    if token.is_punct:
        print(token,'is a punctuation token\nUsed ".is_punct" field to determine this')

print('Checking whether the token is number or not')
for token in doc:
    # print(token,'--->',token.like_num)
    if token.like_num:
        print(token,'is a number\nUsed ".like_num" field to determine this')


table = PrettyTable()
table.field_names=['token','is_alpha','is_punct','like_num']
for token in doc:
    table.add_row([token.text,token.is_alpha,token.is_punct,token.like_num])

print(table)