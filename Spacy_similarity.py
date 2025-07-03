import spacy
from prettytable import PrettyTable

# Sentence similarity is dependent on projecting your text into vector space, we use this vector in vector embeddings and other stuff later but for now the text data is being projected into vector data
# Similar text will be having same kind of vectors


'''
1. Similarity in NLP quantifies how closely related two pieces of text are, based on the factors like word choice, context, and meaning
2. USED IN VARIOUS APPLICATIONS : Crucial for tasks such as document clustering, recommendation systems, and paraphrase detection, where understanding the degree of similarity between texts is essential 
3. DIFFERENT METHODS: Similarity can be measured using various approaches, from simple techniques like cosine similarity on word vectors to more advanced methods involving contextual embeddings from models like BERT.

'''
nlp = spacy.load('en_core_web_lg')
doc = nlp('cat')
vec1 = doc.vector
print(doc.vector_norm)
print(doc.vector.shape)
#print(doc.vector) # Gives an array

doc = nlp('dog')
vec2 = doc.vector
print(doc.vector_norm)
print(doc.vector.shape)

'''
Since we got two vectors, we can use cosine similarity to check how relatable these two vectors are

TO USE THE COSINE SIMILARITY WE HAVE TO IMPORT FROM SKLEARN.METRICS.PAIRWISE IMPORT COSINE_SIMILARITY
'''
from sklearn.metrics.pairwise import cosine_similarity

sim = cosine_similarity([vec1],[vec2])
print(sim)# 80% similar

words = ['dog','cat','king','queen','man','women','houses']
table = PrettyTable(['word 1','word 2','similarity score'])

from itertools import combinations
print(list(combinations(words,2)))# Has all the combinations in a given list of words

for word1, word2 in combinations(words,2):
    token1 = nlp(word1)
    token2 = nlp(word2)

    sim = token1.similarity(token2)
    table.add_row([token1.text,token2.text,f'{sim:.2f}'])
print(table)