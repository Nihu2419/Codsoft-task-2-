import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tokenize import word_tokenize
import sklearn.metrics as m
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
nltk.download('stopwords')
nltk.download('punkt')
dataset=pd.read_csv('F:\spamsms\spam.csv',encoding='latin-1')
dataset
sent=dataset.iloc[:,[1]]['v2']
sent
label=dataset.iloc[:,[0]]['v1']
label
import re
len(set(stopwords.words('english')))
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
label=le.fit_transform(label)
label
lemma=WordNetLemmatizer()
sent
for sen in sent:
  senti=re.sub('[^A-Za-z]',' ',sen)
  senti=senti.lower()
  words=word_tokenize(senti)
  word=[lemma.lemmatize(i) for i in words if i not in stopwords.words('english')]
  senti=' '.join(word)
  sentences.append(senti)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(max_features=5000)
features=tfidf.fit_transform(sentences)
features=features.toarray()
features
feature_train,feature_test,label_train,label_test=train_test_split(features,label,test_size=0.2,random_state=7)
model=MultinomialNB()
model.fit(feature_train,label_train)
label_pred=model.predict(feature_test)
label_pred
label_test
m.accuracy_score(label_test,label_pred)
print(m.classification_report(label_test,label_pred))
print(m.confusion_matrix(label_test,label_pred))
model=SVC(kernel='linear')
model.fit(feature_train,label_train)
label_pred=model.predict(feature_test)
m.accuracy_score(label_test,label_pred)
label_pred
label_test
print(m.classification_report(label_test,label_pred))
print(m.confusion_matrix(label_test,label_pred))
model=LogisticRegression()
model.fit(feature_train,label_train)
label_pred=model.predict(feature_test)
m.accuracy_score(label_test,label_pred)
label_pred
label_test
print(m.classification_report(label_test,label_pred))
print(m.confusion_matrix(label_test,label_pred))
model=DecisionTreeClassifier()
model.fit(feature_train,label_train)
label_pred=model.predict(feature_test)
m.accuracy_score(label_test,label_pred)
label_pred
label_test
print(m.classification_report(label_test,label_pred))
print(m.confusion_matrix(label_test,label_pred))