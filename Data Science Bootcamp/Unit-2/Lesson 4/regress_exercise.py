import pandas as pd
import numpy as np
import math
import warnings
from IPython.display import display
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import statsmodels.formula.api as smf

path ='https://raw.githubusercontent.com/Thinkful-Ed/data-201-resources/master/New_York_offenses/NEW_YORK-Offenses_Known_to_Law_Enforcement_by_City_2013%20-%2013tbl8ny.csv'
crime = pd.read_csv(path,header=4)
crime.drop(columns=['Rape\n(revised\ndefinition)1'],axis=1)
int(crime.loc[10]['Population'].replace(',',''))

crime['Population'] = crime['Population'].apply(lambda x: float(str(x).replace(',','')))
crime['Population_2'] = crime['Population'].apply(lambda x: float(str(x).replace(',',''))**2)
crime['Robbery']= crime['Robbery'].apply(lambda x: str(x).replace('\'\'',''))
crime['Robbery']= crime['Robbery'].apply(lambda x: float(str(x).replace(',','')))
crime['Murder'] = np.where(crime[crime.columns[3]]>0, 1, 0)
crime['Robbery']= np.where(crime['Robbery']>0, 1, 0)

crime['Population'].idxmax(axis=0)
features = pd.DataFrame()
features[['Population','Population_2','Robbery','Murder']] = crime[['Population','Population_2','Robbery','Murder']]
features = features.drop(features['Population'].idxmax(axis=0))
features[['Population','Population_2','Robbery','Murder']].boxplot()
features[['Population','Population_2','Robbery','Murder']].describe()


regr = linear_model.LinearRegression()
X = features[['Population','Population_2','Robbery','Murder']]
features['Property']= crime[crime.columns[8]].apply(lambda x: float(str(x).replace(',','')))

X_temp = X.dropna(axis=0,how='any')
Y_temp = features['Property']



zer = features[features['Property']==0]
X_feat1 = X_temp.drop(zer.index[0:len(zer)])
Y_temp = features['Property'].drop(zer.index[0:len(zer)])
Y_temp = Y_temp.dropna(axis=0,how='any')

Y_feat1 = np.log(Y_temp)
Y_feat2 = np.sqrt(Y_temp)
regr.fit(X_feat1, Y_feat1)

# Inspect the results.
print('\nCoefficients: \n', regr.coef_)
print('\nIntercept: \n', regr.intercept_)
print('\nR-squared:')
print(regr.score(X_feat1, Y_feat1))
# Extract predicted values.
predicted = regr.predict(X_feat1).ravel()
actual = Y_feat1
# Calculate the error, also called the residual.
residual = actual - predicted

# This looks a bit concerning.
plt.hist(residual)
plt.title('Residual counts')
plt.xlabel('Residual')
plt.ylabel('Count')
plt.show()

plt.scatter(predicted, residual)
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.axhline(y=0)
plt.title('Residual vs. Predicted')
plt.show()

regr.fit(X_feat1, Y_feat2)

# Inspect the results.
print('\nCoefficients: \n', regr.coef_)
print('\nIntercept: \n', regr.intercept_)
print('\nR-squared:')
print(regr.score(X_feat1, Y_feat2))
# Extract predicted values.
predicted = regr.predict(X_feat1).ravel()
actual = Y_feat2
# Calculate the error, also called the residual.
residual = actual - predicted

# This looks a bit concerning.
plt.hist(residual)
plt.title('Residual counts')
plt.xlabel('Residual')
plt.ylabel('Count')
plt.show()

plt.scatter(predicted, residual)
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.axhline(y=0)
plt.title('Residual vs. Predicted')
plt.show()


features_temp = features.dropna(axis=0,how='any')
zer = features[features['Property']==0]
features_temp = features_temp.drop(zer.index[0:len(zer)])
features_temp = features_temp.loc[0:len(features_temp)]
X_new = features_temp[['Population','Population_2','Robbery','Murder']]
Y_new = np.log(features_temp['Property'])
regr.fit(X_new,Y_new)
print('\nCoefficients: \n', regr.coef_)
print('\nIntercept: \n', regr.intercept_)
print('\nR-squared:')
print(regr.score(X_new, Y_new))
# Extract predicted values.
predicted = regr.predict(X_new).ravel()
actual = Y_new
# Calculate the error, also called the residual.
residual = actual - predicted

# This looks a bit concerning.
plt.hist(residual)
plt.title('Residual counts')
plt.xlabel('Residual')
plt.ylabel('Count')
plt.show()

plt.scatter(predicted, residual)
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.axhline(y=0)
plt.title('Residual vs. Predicted')
plt.show()
