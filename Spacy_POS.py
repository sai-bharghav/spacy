import spacy
from prettytable import PrettyTable
from spacy import displacy

nlp = spacy.load('en_core_web_md')

doc = nlp("Hello Bharghav, try to be good and learn spacy.You are 26 years old. You can't be that lazy")

# POS tells how a particular used in that specific sentence
'''
1. Understanding Sentence Structure:
POS Tagging helps identify the grammatical structure of a sentence by categorizing words into their respective parts of speech(nouns, verbs, etc)
This understanding is crucial for tasks like synthetic parsing, where relationship between words is analyzed

2. Enhancing NLP Tasks
Many NLP Tasks, such as named entity recognition, text classification, and machine translation, benefit from POS tagging. It provides additional
context that can improve the accuracy and performance of these models

3. Word Sense Disambiguation
POS Tagging aids in word sense disambiguation by providing context abut a word's role in a sentence,, helping distinguish between meanings
of a word depending on its parts of speech.

'''

table = PrettyTable()
table.field_names = ['token','pos','details','explanation']
for token in doc:
    table.add_row([token,token.pos_,token.tag_,spacy.explain(token.tag_)])

print(table)

# DEPENDENCY PARSING
# Use dependency parsing to find the grammatical relationships between words and understand sentence structure.
# What: Dependency parsing identifies the "boss" (head word) for each word in the sentence.
# Does: This lets us understand "who did what to whom" by linking subjects, verbs, and objects.
'''
In dependency token every sentence is like a tree where the root word(token) is determined and rest all the other words(tokens) are dependent tokens
'''

# POS Tagging focusses on relationships and is flat labelling while 'Dependency parsing' is something that focus on the relationships of the head word to the dependent words and has a hierarchial structure
# Let us use the code
table = PrettyTable()
table.field_names = ['token','dependency','head word','head POS','children']

doc = nlp('The dead air shaped the dead darkness, further away than seeing shapes the dead earth.')
for token in doc:
    #Since for every token the children is different, if we want to have a look at the children we have to store it in a variable and use the 'for' loop below
    children = [child.text for child in token.children]
    table.add_row([token.text,token.dep_,token.head.text,token.head.pos_,children])

print(table)

# Let us use the displacy from spacy
displacy.render(doc, style='dep',options={},jupyter=True)