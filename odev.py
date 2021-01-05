import pandas as pd
import string
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def nGram(text, n): #n degerinde n-gram alınıyor
    ngrams(text.split(), n)
    return ngrams

def removePunc(text): #noktalama isaretleri temizleme
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text
    
def textTokenizer(text):
    return word_tokenize(text)

def toLower(text):
    return text.lower()

def stpWords(text):
    stops = set(stopwords.words('english'))
    text = " ".join([w for w in text if not w in stops])
    return text
#text = removePunc("enver sanli sivas yolcusu,  yakinda.")
#print(text)
#textTokenizer(text)


data = pd.read_csv("data.csv") #Train data yüklendi
#test_df = pd.read_csv('test.csv')  #test etmek için test data yüklendi 

data['author_num'] = data.author.map({'EAP':0, 'HPL':1, 'MWS':2})
#print(data.shape)
X = data['text']
y = data['author_num']

#line =(data['text']) #text kolonu line dizisine basıldı.


#for text in data['text']:
#    text = text.lower()
#    print(text)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=109)
print("--------------------------")
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

print("--------------------------")



data_len = (len(X_train) - 1)
i = 1

while i < 10:
    #X_train[i] = X_train[i].lower()
    #X_train[i] = removePunc(X_train[i])
    #X_train[i] = textTokenizer(X_train[i])
   
    #print(X_train[int(i)])
    i = i + 1
  
X_train = X_train.apply(lambda x: toLower(x))
X_train = X_train.apply(lambda x: removePunc(x))
X_train = X_train.apply(lambda x: textTokenizer(x))
X_train = X_train.apply(lambda x: stpWords(x))
 
#print(X_train)
vectorizer=TfidfVectorizer(max_features=1000,
                           use_idf=False,
                           lowercase=False,
                           norm=None,
                           ngram_range=(1,1))

X_train = vectorizer.fit_transform(X_train)
X_train = X_train.toarray()

print(X_train.shape)
#Test datası için
X_test = X_test.apply(lambda x: toLower(x))
X_test = X_test.apply(lambda x: removePunc(x))
X_test = X_test.apply(lambda x: textTokenizer(x))
X_test = X_test.apply(lambda x: stpWords(x))

X_test = vectorizer.transform(X_test)

X_test = X_test.toarray()
print(X_test.shape)

model = GaussianNB()

y_pred = model.predict(X_test)

print(y_pred)
#res = vectorizor.fit_transform(X_train)
#print(res)
#print(line.lower()) #bütün harfler küçüğe çevrildi




#model.fit(X_train, y_train)
#y_pred = model.predict(X_test)
#print(y_pred)
#print(line[0].translate(str.maketrans('', '', string.punctuation)))


