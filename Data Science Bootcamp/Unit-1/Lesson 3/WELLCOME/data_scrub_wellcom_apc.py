import pandas as pd
import pkgutil
import encodings
import os
import csv
import matplotlib.pyplot as plt

path =  'C:/Users/clyde/Documents/Thinkful/Data Science Bootcamp/Unit-1/Lesson 3/WELLCOME'
data = pd.DataFrame()

with open(path+'/WELLCOM_APC.csv', 'r', encoding='ascii', errors='ignore') as infile:
     inputs = csv.reader(infile)
     for index, row in enumerate(inputs):
         data[index] =''
         data[index]= row

data = data.T
data.columns = [data.loc[0][0],data.loc[0][1],data.loc[0][2],data.loc[0][3],data.loc[0][4]]
data = data.drop([0])
data_na = data[data.loc[:]['PMID/PMCID']=='NA']
data = data.drop(data_na.index)
data_journ_cnt = data.groupby('Journal title').count()
data_journ_cnts = data_journ_cnt.sort_values(by = 'Article title')

data_journ_cnt = data.groupby('Journal title')[data.columns[4]].count()
data_journ_cnt = data_journ_cnt.sort_values()
data_journ_cnt.tail(5).plot.bar(rot=90,figsize=(9,5))
plt.show()

#data_journ_sum = pd.to_numeric(data.groupby('Journal title')[data.columns[4]],errors ='ignore')
data_journ_mean = data.groupby('Journal title')[data.columns[4]].agg(lambda x: pd.to_numeric(x, errors='coerce').mean())
data_journ_median = data.groupby('Journal title')[data.columns[4]].agg(lambda x: pd.to_numeric(x, errors='coerce').median())
data_journ_std = data.groupby('Journal title')[data.columns[4]].agg(lambda x: pd.to_numeric(x, errors='coerce').std())

data_convert_cost = data.groupby(data.columns[4]).agg(lambda x: pd.to_numeric(x, errors='coerce'))