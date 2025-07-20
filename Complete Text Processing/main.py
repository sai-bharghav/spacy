'''
GENERAL FEATURE EXTRACTION 
- File loading
- Word Counts
- Character count
- Average characters per word
- Stop words count
- Count #Hashtag and @Mentions
- If numeric digits are present in tweet
- Upper case word counts


PREPROCESSING AND CLEANING
- Lower case
- Contraction to Expansion
- Email removals and count
- URLs removal and counts
- Removal of RT
- Removal of Special Characters
- Removal of multiple spaces
- Removal of HTML tags
- Removal of accented characters
- Removal of stop words
- Conversion into base form of words
- Common occuring words removal
- Rare Occuring words removal
- Word cloud
- Spelling correction 
- Tokenization 
- Lemmatization 
- Detecting entities using NER
- Noun Detection 
- N-gram, word count, Lemmatization and POS tagging
- Using inbuilt sentiment classifier
- Sentence Translation 
- Language Detection

'''




## A. General Feature Extraction 

#### Data Loader
import pandas as pd 
import numpy as np 
import re

df = pd.read_csv('.\\Complete Text Processing\\twitter4000.csv')
print(df.head()) #Checking the first 5 rows of the dataframe
print('--'*70)
print(df.info())# Checking the info on the dataframe(basic) to get an idea about the dtype of the columns 
print('--'*70)
print(df.isnull().sum())# Checking whether there are any null values in the dataframe
print('--'*70)

print('The count of each type of sentiment is given below')
print(df['sentiment'].value_counts())

## B. Characters Count
# Check for the count of characters for each row on tweets column, for that let's use regex and lambda function.
pattern = r'\s' # Pattern to find spaces so that we can exclude them and then count only the characters
print('--'*70)
print('Getting only the character count without any spaces(using regex pattern \\s and using re.sub(pattern, '', text) and having it as a column)')
print(df['tweets'].apply(lambda x: len(re.sub(pattern,'',x))))# using sub from regex to detect the pattern in a text(x) and then replacing it with ''(no space)
df['char_counts']= df['tweets'].apply(lambda x: len(re.sub(pattern,'',x)))
# print(df.sample(5))

## 3. Word Counts
# Let us count the number of words in the tweets 
print('--'*70)
print('Creating a new column word_counts to count the number of words in a tweet and storing it in a column')
print(df['tweets'].apply(lambda x: len(x.split())))
df['word_counts'] = df['tweets'].apply(lambda x: len(x.split()))

## 4. Average Word Length
print('--'*70)
print('Creating a column avg_word_len (Having the length of the tweets and making a column for it)')
df['avg_word_len'] = df['char_counts']/df['word_counts']
df['avg_word_len'] = df['avg_word_len'].apply(lambda x: round(x,1))
print(df.sample(5))


## 5. Stop Words Count
from spacy.lang.en.stop_words import STOP_WORDS as stopwords ## A bit different compared to nlp.Defaults.stop_words
# It is the same stopwords as nlp.Defaults.stop_words since the length of both is the same(326)
print('--'*70)
print('Getting the length of the stop words in the tweets column (taking stopwords from spacy and using lambda function to apply it to whole column)')
print(df['tweets'].apply(lambda x:len([word for word in x.lower().split() if word in stopwords]))) ## ALWAYS REMEMBER TO CONVERT EVERYTHING INTO LOWER CASE
df['stop_words_len']=df['tweets'].apply(lambda x:len([word for word in x.lower().split() if word in stopwords]))
print('Seeing the sample of the dataframe')


## 6. Count Hashtags and Mentions
print("--"*70)
print('Creating a column hashtag_count and mentions_count in the dataframe suing regex')
df['hashtag_counts']=df['tweets'].apply(lambda x:len(re.findall(r'#\w+',x)))# Using re.findall() we give the pattern saying to get the word after '#' character
df['mentions_count']=df['tweets'].apply(lambda x:len(re.findall(r'@\w+',x)))# Using re.findall() we give the pattern saying to get the word after '@' character


print('Check the rows which have hastag_count greater than 0 or mentions_count greater than 0')
print('--'*70)



## 7. If numeric digits present in tweets
print('--'*70)
print('Checking whether if integer data is present in tweets')
# Using re.findall() we can get the match and then we can count the number of times it is present in tweets
# Checking on the integer data, considering only the word 10 instead of ABC232, where I only want 10 but not ABC232
df['numeric_count']=df['tweets'].apply(lambda x:len(re.findall(r'\b\d+\b',x)))


## 8. UPPER case words count
print('--'*70)
print('Counting the Upper case words count')
# Straight forward code check whether the word in a tweet is upper case and then count the number of words 
df['uppercase_count']= df['tweets'].apply(lambda x:len([word for word in x.split() if x.isupper()]))


### The first part A(General Feature Extraction is done)

# B PREPROCESSING AND CLEANING 
## 9 Lower case Conversion


# Some of the tweets might have more than two spaces so we will using it to convert it into one space and then apply lower method
# Since we are preprocessing the data and not extracting any features, we will apply the changes to the tweets column 
print('--'*70)
df['tweets'] = df['tweets'].apply(lambda x: re.sub('\s+',' ',x).lower())


## 10. Contraction to Expansion
# The contractions.json file is required since we will be dealing with words let's and it's full form is let us, where it has the all those type of words in it.
import json 
contractions = json.load(open('contractions.json'))

# Every contraction word is in lower case so it might not get the values such I'm where 'I' is in capital form. We have to convert these into lower form.
# Let us modify the tweets where the contracted words are in the full form

df['tweets'] = df['tweets'].apply(lambda x :" ".join([contractions.get(word.lower(),word) for word in x.split()]))
# Here we are splitting the word and converting it into lower form, then we are getting the value of that word if it is contracted or we are giving it default value. IT IS THE REASON WHY WE ARE DOING contractions.get(word.lower(), word) where the second "word" is for default value

## 11. Count and Remove Emails 
print("--"*70)
# we will use regex pattern to find the emails in the tweets column (choll@gmail.com, choll@gmail.co.in)
pattern = r'\b[A-Za-z0-9_%+-]+@[A-Za-z0-9-.]+\.[A-Z|a-z]{2,}\b' # Make sure that you use '-' only at the end if you are trying to find the character since '-' can also be interpreted as range.
# We will use re.findall() method to find the emails and then apply it using a lambda function 
# Creating a new columns 'emails' in the dataframe to get the emails and joining it with ","

df['emails']=df['tweets'].apply(lambda x: ",".join(re.findall(pattern=pattern,string=x)))
print('Counting the number of emails we have in the dataframe')
print(df['emails'].value_counts())
# Now since we are having ',' in the emails column like (email1, email2) we can use the count of ',' and then get the count in different column 'email_counts'
df['email_counts'] = df['emails'].apply(lambda x:x.count(',')+1 if len(x)>0 else 0)
# Let us take out the emails from the tweet and replace it with normal '' 
df['tweets'] = df['tweets'].apply(lambda x : re.sub(pattern,'',x))


## 12. Count and Remove URLs
# If there is any URL in the tweets column, count them and remove them from the tweets
# We might have hhtps:google.com or www.google.com, the pattern is 
print("--"*70)
pattern = r'http\S+ | www\.\S+'
# Creating a new column to get the urls in a list 
df['urls'] = df['tweets'].apply(lambda x : re.findall(pattern,x))
# Getting the value counts of each urls 
print('The count of each urls')
print(df['urls'].value_counts())
#Creating a new column for the count of urls 
df['urls_counts'] = df['urls'].apply(lambda x: len(x))

# Since we got the urls from the tweets, it is time to remove them from the tweets column 
print('Removing the urls from the tweets column ')
df['tweets'] = df['tweets'].apply(lambda x : re.sub(pattern, '',x))


## 13. Remove RT
# Remove the retweets 
print("--"*70)
pattern = r'\bRT @\w+' # RT means here is the retweet
print('Checking whether a tweet has a retweet')
df['is_retweet'] = df['tweets'].apply(lambda x: bool(len(re.findall(pattern,x))))

# There is no retweets in the data, if we have any retweets, here is how you can remove them
print('Removing the retweets')
df['tweets']=df['tweets'].apply(lambda x: re.sub(pattern, '',x))


# 14. Remove HTML tags if present in the data(Tweets)
from bs4 import BeautifulSoup # you also need to install lxml parcel since we will be using it in beautifulsoup
print("--"*70)
# Removing the tags if present
print('Removing HTML tags(p) if present and getting only the text')
df['tweets'] = df['tweets'].apply(lambda x: BeautifulSoup('<p>'+x+'</p>','lxml').get_text())

#15. Remove Accented Characters
## What is an Accented Characters
## These are a special characters in english where it is derived from ancient greek language
import unicodedata
print('--'*70)
print('Normalizing the tweets column to replace the accented characters with it\'s adjacent characters in ascii')
df['tweets'] = df['tweets'].apply(lambda x: unicodedata.normalize('NFKD',x).encode('ascii','ignore').decode('utf-8','ignore'))



## 16. Special Chars removal and punctuation removal 
# Removing the @,# or any other special characters mentioned in the tweets column using regex and the pattern is below
pattern = r'@\w+'# This one is for removing
print('--'*70)
print('Removing mentions in the tweets column')
df['tweets'] = df['tweets'].apply(lambda x :re.sub(pattern,'',x).strip())# .strip() is or removing all the leading or lagging spaces after subbing 
print('Removing the special characters using regex')
pattern = r'[^\w\s]' #this one is only for removing special characters
df['tweets'] = df['tweets'].apply(lambda x :re.sub(pattern,'',x).strip())

## 17. Remove Repeated Characters
# We might have so much repeated characters in the data where we will be having like 'awwwwwwwwwwwwww' where we have to remove it and make it single to 'aw
print('--'*70)
pattern = r'(.)\1+'

print('Removing extra characters (repeated)')
df['tweets'] = df['tweets'].apply(lambda x: re.sub(pattern, r'\1\1',x))# \1\1 is for replacing awwwwwwwww into aww(2'w')


## 18. Remove Stop words 
print('--'*70)
print('Removing the stop words in the tweets column and storing it in a column')# Creating a column tweets_no_stop_words
df['tweets_no_stop_words']= df['tweets'].apply(lambda x: " ".join([word for word in x.split() if word not in stopwords]))


## 19. Convert the base or root form of word
import spacy

nlp = spacy.load('en_core_web_sm') # Load the small model of spacy, you can also use en_core_web_md or en_core_web_lg for medium and large models respectively
# If you don't have the model, you can download it using the command: python -m spacy download en_core_web_sm

# Let us create a function for lemmatizing each word where you only lemmatize the word if the word is NOUN or VERB
def lemmatize_noun_verb(x):
    doc = nlp(x)
    tokens = []
    for token in doc:
        if token.pos_ in ['NOUN','VERB']:
            tokens.append(token.lemma_)
        else:
            tokens.append(token.text)
    x = " ".join(tokens)
    pattern = r'\s\.'
    x = re.sub(pattern,'.',x)
    return x

print('Converting the NOUN and VERB and storing it in a "base_tweets" column')

df['base_tweets']=df['tweets'].apply(lambda x:lemmatize_noun_verb(x))


## 20. Common words removal 
# Let me get the common words on the column with no stopwords
# Now let us import counter to count the number of words 

from collections import Counter
# Convert all tweets with no stopwords into a sentence and store it
x = ' '.join(df['tweets_no_stop_words'].tolist())
words = x.split()

word_freq = Counter(words)

top10 =[word for word in word_freq.most_common(10)]# Getting the top 10 words
print('Creating a column with no common words in the tweet')

df['no_common_tweets'] = df['tweets'].apply(lambda x : ' '.join([word for word in x.split() if word not in top10]))
print(df)

## 21. Rare words Removal
# Let us take out the last 10 rare words 

least10 = [word[0] for word in word_freq.most_common()[-11:]]
print('Creating a column with no common least words in the tweet')

df['no_least_tweets'] = df['tweets'].apply(lambda x: ' '.join([word for word in x.split() if word not in least10]))


## 22. Word Cloud Visualization 
#pip install wordcloud
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# x = ' '.join(df['tweets_no_stop_words'].to_list())
# cloud = WordCloud(width=500,height=500).generate(x)

# print('Showing Word Cloud')
# plt.figure(figsize=(10,5))
# plt.imshow(cloud,interpolation='bilinear')
# plt.colorbar() 
# plt.show()

## 23. Spelling Correction 
# pip install -U textblob==0.18.0.post0
# python -m textblob.download_corpora

from textblob import Word

'''
The textblob has a good way of finding the right spelling based on probabilities, we can say 
x='Loooove' and when you type w=Word(x) and x.spellcheck() it give the right word along with probabilities

In the below method, where a particular sentence is sent as an argument where we are creating a dummy list 'words' and the input argument
will be split into spaces(converting into lists) then making sure to append the right word of each word(w) and joining it later
'''

def correct(x):
    words =[] 
    for word in x.split():
        w = Word(x)
        words.append(w.correct())

    return ' '.join(words)

# Now we can apply the correct function to the tweets column
print('--'*70)
print('Correcting the spelling of the tweets column for 10 samples')

data=df.sample(10,random_state=0)
data['correct_tweets'] = data['tweets'].apply(lambda x: correct(x))
print(data[data['tweets']!=data['correct_tweets']])