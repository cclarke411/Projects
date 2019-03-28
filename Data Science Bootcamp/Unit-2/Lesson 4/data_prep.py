import pandas as pd
import numpy as np
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