import seaborn as sns
import pandas as pd
import wordcloud
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB
import numpy as np


amazon_data = pd.DataFrame()
cnt = 0
with open('amazon_cells_labelled.txt') as amazon_file:
    text = amazon_file.readlines()

data1 = {'data':text}
df = pd.DataFrame(data1, columns = ['data'])
df1 = pd.DataFrame()
#[df.loc[i]['indent'] = df.loc[i]['data'].split('\t')[1].split('\n')[0] for i,x in enumerate(df)]

df1[['Data','Indent']] = df.data.str.split("\t",expand=True,)
df1['Value'] = df1['Indent'].str.extract('(\d)', expand=True)

temp = pd.DataFrame()
temp['data'] = df1['Data']
temp['count'] = df1['Data'].apply(lambda x: len(str(x).split(" ")))
temp['char_count'] = df1['Data'].str.len()

def avg_word(sentence):
  words = sentence.split()
  return (sum(len(word) for word in words)/len(words))

temp['avg_word'] = df1['Data'].apply(lambda x: avg_word(x))

from nltk.corpus import stopwords
stop = stopwords.words('english')
temp['stopwords'] = df1['Data'].apply(lambda x: len([x for x in x.split() if x in stop]))

temp['hastags'] = df1['Data'].apply(lambda x: len([x for x in x.split() if x.endswith('!')]))

temp['numerics'] = df1['Data'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

temp['upper'] = df1['Data'].apply(lambda x: len([x for x in x.split() if x.isupper()]))

temp['great'] = np.where((df['data'].str.contains('great', regex=True)), 1, 0)
temp['good'] = np.where((df['data'].str.contains('good', regex=True)), 1, 0)
temp['excellent'] = np.where((df['data'].str.contains('excellent', regex=True)), 1, 0)

temp['time'] = np.where((df['data'].str.contains('time', regex=True)), 1, 0)


amazon = temp.loc[:,['stopwords','numerics','upper','avg_word','count','char_count']]
target = df1['Value']
amazon = amazon.dropna()
# Instantiate our model and store it in a new variable.
bnb = BernoulliNB()

# Fit our model to the data.
bnb.fit(amazon, target)

# Classify, storing the result in a new variable.
y_pred = bnb.predict(amazon)

# Display our results.
print("Number of mislabeled points out of a total {} points : {}".format(
    amazon.shape[0], 
    (target != y_pred).sum()))

# Build your confusion matrix and calculate sensitivity and specificity here.
zz = (y_pred==target).sum()
xx = (y_pred!=target).sum()
qq = ((target == False) & (y_pred ==  False)).sum()
ww = ((target == False) & (y_pred ==  True)).sum()
tt = ((target == True)  & (y_pred ==  False)).sum()
uu = ((target == True ) & (y_pred ==  True)).sum()
print(qq,ww,tt,uu)
vv = np.array([[qq,ww],[tt,uu]])
vv