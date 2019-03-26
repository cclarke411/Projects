import pandas as pd
import numpy as np
import seaborn as sns
import wordcloud
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from sklearn.naive_bayes import BernoulliNB


yelp_data = pd.DataFrame()
with open('yelp_labelled.txt') as amazon_file:
    text = amazon_file.readlines()

data1 = {'data':text}
dfy = pd.DataFrame(data1, columns = ['data'])
dfy1 = pd.DataFrame()
#[df.loc[i]['indent'] = df.loc[i]['data'].split('\t')[1].split('\n')[0] for i,x in enumerate(df)]

dfy1[['Data','Indent']] = dfy.data.str.split("\t",expand=True,)
dfy1['Value'] = dfy1['Indent'].str.extract('(\d)', expand=True)


dfy1['Value'] = dfy1['Value'].apply(float)
dfy1_pos = dfy1[dfy1['Value'] ==1]
dfy1_neg = dfy1[dfy1['Value'] ==0]

yelp_pos_words = ''
yelp_neg_words = ''

for i in dfy1_pos['Data'].index:
    yelp_pos_words = yelp_pos_words + dfy1_pos.loc[i]['Data']
    
for i in dfy1_neg['Data'].index:   
    yelp_neg_words = yelp_neg_words + dfy1_neg.loc[i]['Data']

pos_words = WordCloud().generate(yelp_pos_words)
neg_words = WordCloud().generate(yelp_neg_words)

plt.imshow(pos_words, interpolation='bilinear')
plt.axis("off")
plt.title("Good")
plt.show()

plt.imshow(neg_words,interpolation = 'bilinear')
plt.axis("off")
plt.title("Bad")
plt.show()
features = pd.DataFrame()

features['great'] = np.where((df['data'].str.contains('great', regex=True)), 1, 0)
features['good'] = np.where((df['data'].str.contains('good', regex=True)), 1, 0)
features['excellent'] = np.where((df['data'].str.contains('excellent', regex=True)), 1, 0)

features['time'] = np.where((df['data'].str.contains('time', regex=True)), 1, 0)

features['phone'] = np.where((df['data'].str.contains('phone', regex=True)), 1, 0)
features['product'] = np.where((df['data'].str.contains('product',regex=True)),1,0)
features['work'] = np.where((df['data'].str.contains('work',regex=True)),1,0)

features['best'] = np.where((df['data'].str.contains('best',regex=True)),1,0)
features['happy'] = np.where((df['data'].str.contains('happy',regex=True)),1,0)
features['price'] = np.where((df['data'].str.contains('price',regex=True)),1,0)

features['quality'] = np.where((df['data'].str.contains('quality',regex=True)),1,0)
features['nice'] = np.where((df['data'].str.contains('nice',regex=True)),1,0)


#TEST FEATURES
features['best_happy_price'] = features['best']+features['happy']+features['price']
features['great_good_excellent'] = features['great']+features['good']+features['excellent']
features['product_work_phone']= features['product']+features['work']+features['phone']
features['value'] = df1['Value']
features = features.dropna()

features['best_happy_price'] = (features['best_happy_price'] - features['best_happy_price'].mean())/features['best_happy_price'].std()
features['great_good_excellent'] = (features['great_good_excellent'] - features['great_good_excellent'].mean())/features['great_good_excellent'].std()
features['product_work_phone'] = (features['product_work_phone']-features['product_work_phone'].mean())/features['product_work_phone'].std()
corrmat = features.corr()

# Set up the matplotlib figure.
f, ax = plt.subplots(figsize=(12, 9))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()

amazon = features.loc[:,['great_good_excellent','product_work_phone','time','best_happy_price','quality','nice']]
target = features['value']

# Instantiate our model and store it in a new variable.
bnb = BernoulliNB()

# Fit our model to the data.
bnb.fit(amazon, target)

# Classify, storing the result in a new variable.
y_pred = bnb.predict(amazon)

# Display our results.
print("Number of mislabeled points out of a total {} points : {}".format(
    amazon.shape[0], 1000 -
    (target != y_pred).sum()))


