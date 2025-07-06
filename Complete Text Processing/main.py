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
print(df.sample(5))

## 6. Count Hashtags and Mentions
print("--"*70)
print('Creating a column hashtag_count and mentions_count in the dataframe suing regex')
df['hashtag_counts']=df['tweets'].apply(lambda x:len(re.findall(r'#\w+',x)))# Using re.findall() we give the pattern saying to get the word after '#' character
df['mentions_count']=df['tweets'].apply(lambda x:len(re.findall(r'@\w+',x)))# Using re.findall() we give the pattern saying to get the word after '@' character
print(df.head())

print('Check the rows which have hastag_count greater than 0 or mentions_count greater than 0')
print('--'*70)
print(df[(df['hashtag_counts']>0) | (df['mentions_count']>0) ])


## 7. If numeric digits present in tweets
print('--'*70)
print('Checking whether if integer data is present in tweets')
# Using re.findall() we can get the match and then we can count the number of times it is present in tweets
# Checking on the integer data, considering only the word 10 instead of ABC232, where I only want 10 but not ABC232
df['numeric_count']=df['tweets'].apply(lambda x:len(re.findall(r'\b\d+\b',x)))
print(df.sample(5))

## 8. UPPER case words count
print('--'*70)
print('Counting the Upper case words count')
# Straight forward code check whether the word in a tweet is upper case and then count the number of words 
df['uppercase_count']= df['tweets'].apply(lambda x:len([word for word in x.split() if x.isupper()]))
print(df[df['uppercase_count']>1].sample(5))

### The first part A(General Feature Extraction is done)

# B PREPROCESSING AND CLEANING 
## 9 Lower case Conversion


# Some of the tweets might have more than two spaces so we will using it to convert it into one space and then apply lower method
# Since we are preprocessing the data and not extracting any features, we will apply the changes to the tweets column 
print('--'*70)
df['tweets'] = df['tweets'].apply(lambda x: re.sub('\s+',' ',x).lower())
print(df.sample(5))

## 10. Contraction to Expansion

