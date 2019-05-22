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
crime_fl = pd.read_csv(path,header=4)
path ='C:/Users/clyde/Documents/Thinkful/Data Science Bootcamp/Unit-2/Lesson 4/table_8_offenses_known_to_law_enforcement_new_york_by_city_2013.csv'
crime_ny = pd.read_csv(path,header=4)
crime = pd.concat([crime_fl,crime_ny],sort='False')
crime = crime.drop(columns=['Rape\n(revised\ndefinition)1'],axis=1)
crime = crime.drop(columns =['Unnamed: 13'])
#int(crime.loc[10]['Population'].replace(',',''))

crime['Population'] = crime['Population'].apply(lambda x: float(str(x).replace(',','')))
crime['Population_2'] = crime['Population'].apply(lambda x: float(str(x).replace(',',''))**2)
crime['Robbery']= crime['Robbery'].apply(lambda x: str(x).replace('\'\'',''))
crime['Robbery']= crime['Robbery'].apply(lambda x: float(str(x).replace(',','')))
crime['Murder'] = np.where(crime[crime.columns[6]]>0, 1, 0)
crime['Robbery']= np.where(crime['Robbery']>0, 1, 0)

crime['Population'].idxmax(axis=0)
features = pd.DataFrame()
features[['Population','Population_2','Robbery','Murder']] = crime[['Population','Population_2','Robbery','Murder']]
features[['Population','Population_2','Robbery','Murder']].boxplot()
features[['Population','Population_2','Robbery','Murder']].describe()

Y = pd.DataFrame()
regr = linear_model.LinearRegression()
X = features[['Population','Population_2','Robbery','Murder']]
Y['Property']= crime[crime.columns[8]].apply(lambda x: float(str(x).replace(',','')))

X_temp = X.dropna(axis=0,how='any')
Y_temp = Y.dropna(axis=0,how='any')
drp_ind =  features['Population'].idxmax(axis=0)

X_temp = X_temp.drop([drp_ind])
Y_temp = Y_temp.drop([drp_ind])

drp_ind1 = Y_temp['Property'].idxmax(axis=0)
X_temp = X_temp.drop([drp_ind1])
Y_temp = Y_temp.drop([drp_ind1])

X_feat1 = X_temp
Y_feat1 = np.log(Y_temp)
Y_feat2 = np.sqrt(Y_temp)
regr.fit(X_temp, Y_feat2)

# Inspect the results.
print('\nCoefficients: \n', regr.coef_)
print('\nIntercept: \n', regr.intercept_)
print('\nR-squared:')
print(regr.score(X_temp, Y_feat2))
# Extract predicted values.
predicted = regr.predict(X_temp).ravel()
actual = Y_temp['Property']
# Calculate the error, also called the residual.
residual = actual - predicted

# This looks a bit concerning.
plt.hist(residual)
plt.title('Residual counts')
plt.xlabel('Residual')
plt.ylabel('Count')
plt.show()

plt.scatter(predicted[residual[residual<4000].index], residual[residual<4000])
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

# fit a model
lm = linear_model.LinearRegression()
model = lm.fit(X_feat1, Y_feat2)
predictions = lm.predict(X_feat1)

## The line / model
plt.scatter(Y_feat2, predictions)
plt.xlabel("True Values")
plt.ylabel("Predictions")

print("Score:", model.score(X_feat1, Y_feat2))


from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics
# Perform 6-fold cross validation
scores = cross_val_score(model, X_feat1, Y_feat2, cv=6)
print('Cross-validated scores:', scores)

# Make cross validated predictions
predictions = cross_val_predict(model, X_feat1, Y_feat2, cv=6)
plt.scatter(Y_feat2, predictions)

accuracy = metrics.r2_score(Y_feat2, predictions)
print('Cross-Predicted Accuracy:', accuracy)
