import spacy
from prettytable import PrettyTable
from spacy import displacy

nlp = spacy.load('en_core_web_md')

'''
What is NER in Simple Words?
Think of Named Entity Recognition (NER) as a smart text highlighter.

Instead of just highlighting words, it scans text to find and classify real-world things—like people, companies, places, dates,
 and money—into predefined categories.
'''
# === spaCy NER Labels: A Quick Reference ===

# --- People, Groups, & Places ---
# PERSON:      People, including fictional characters.
#              (e.g., "Elon Musk", "Harry Potter")

# NORP:        Nationalities or religious or political groups.
#              (e.g., "American", "Christian", "Democrats")

# FAC:         Facilities - buildings, airports, highways, bridges, etc.
#              (e.g., "Eiffel Tower", "DFW Airport", "Interstate 35")

# ORG:         Organizations - companies, agencies, institutions, etc.
#              (e.g., "Google", "FBI", "The University of Texas at Dallas")

# GPE:         Geopolitical Entity - countries, cities, states. These are places with a government.
#              (e.g., "USA", "Texas", "Dallas")

# LOC:         Location - Non-GPE locations, such as mountain ranges, bodies of water.
#              (e.g., "The Rocky Mountains", "Trinity River", "Sahara Desert")


# --- Art, Events, & Products ---
# EVENT:       Named hurricanes, battles, wars, sports events, etc.
#              (e.g., "World War II", "the Olympics", "State Fair of Texas")

# WORK_OF_ART: Titles of books, songs, paintings, movies.
#              (e.g., "The Mona Lisa", "Bohemian Rhapsody", "Hamlet")

# PRODUCT:     Objects, vehicles, foods, etc. (but not services).
#              (e.g., "iPhone 15", "Ford Mustang", "Big Mac")

# LAW:         Named documents made into laws.
#              (e.g., "The Constitution", "The Civil Rights Act")

# LANGUAGE:    Any named language.
#              (e.g., "English", "Spanish", "Python" when referring to the language)


# --- Dates & Numbers ---
# DATE:        Absolute or relative dates or periods.
#              (e.g., "June 26, 2025", "yesterday", "the 21st century")

# TIME:        Time units smaller than a day.
#              (e.g., "4:24 PM", "noon", "a few seconds")

# PERCENT:     Percentage, including "%".
#              (e.g., "20%", "fifty percent")

# MONEY:       Monetary values, including unit.
#              (e.g., "$100", "50 euros", "five dollars")

# QUANTITY:    Measurements, as of weight or distance.
#              (e.g., "10 kg", "25 miles", "a dozen")

# ORDINAL:     "first", "second", "3rd", etc.

# CARDINAL:    Numerals that do not fall under another type.
#              (e.g., "one", "27", "a million")

print('To get all the labels in NER on spacy we can use this code "nlp.get_pipe("ner").labels"')
print(nlp.get_pipe('ner').labels)

text = '''
Apollo Global Management chief economist Torsten Sløk has warned that America is on the verge of a critical inflection point for stagflation, a situation where inflation remains elevated while growth decelerates — something that's particularly challenging to address, according to a Business Insider report. It's different from a typical recession, as stagflation renders the Federal Reserve powerless, making it more difficult to reduce interest rates without further exacerbating inflation, as per the report
'''
doc = nlp(text)
# print('trying to host on http://localhost:5000 so please click the link to go to it')
# displacy.serve(doc,style = 'ent',port=5000)


table = PrettyTable()
table.field_names = ['token','label(NER)','Explanation','Start Char','End Char']


for ent in doc.ents:
    # If you want to loop over the entities specifically, we have to use doc.ents instead of doc 
    table.add_row([ent.text,ent.label_,spacy.explain(ent.label_),ent.start_char,ent.end_char])
    # Labels here are self explanatory .text gives you the text
    # .label_ gives the label of the entity
    # spacy.explain(ent.label_) explains the label, .start_char tells you the start index of the entity character in the text
    # .end_char tells you the end index of the specific entity in the text

print(table)