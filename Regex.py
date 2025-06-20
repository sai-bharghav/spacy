import re
# Ctrl + enter is the shortcut


# BASIC REGEX PYTHON
text = "hello world!!! python is a beautiful language! It is good to learn and would be useful from 2020"
pattern = 'hello'
result = re.match(pattern=pattern, string= text)
print(result)

print('Using group you can get what it matched',result.group())
print('Using span I can get the index where I used .span()',result.span())

pattern = 'python'
result = re.match(pattern=pattern,string=text)
print('Used python without any operators and the result is None---> ',result)



# USING RE.SEARCH()
pattern= 'python'
# text is on the top so no need to do it again
result = re.search(pattern=pattern, string=text)
print('Using .search() the result is',result)

# Searching for the numbers
pattern = '\d+'
result= re.search(pattern=pattern,string=text)
print('Using \\d+ in the pattern which is used for searching for the numbers in the text the result is',result)
'''
If used /d in the pattern without any plus sign(+) like "\d+", it will only select the first digit in the text which is 2 unlike 2025
'''
print(result.group())

# Checking for the email 
text = '''I am studying regex.
' I am a learner where my email is chollangi7@gmail.com, chollangi71@gmail.com, chollangisaibharghav@gmail.com.
I am 25 years old'''
# Basic email pattern 
pattern = '\w+@\w+.\w+'
result = re.search(pattern=pattern, string=text)
print('Basic email search pattern result',result)



#USING RE.FINDALL() TO GET ALL THE MATCHES IN THE TEXT
pattern = '\w+@\w+.\w+'
result = re.findall(pattern=pattern, string=text)
print('Basic email search pattern result using findall() method',result)

#Getting all the digits in the text string instead of getting one
pattern = '\d+'
result = re.findall(pattern=pattern, string = text)
print('Getting all the numbers using findall() method',result)

