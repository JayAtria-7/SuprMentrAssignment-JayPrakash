'''
Todays Task:

create a dataset with columns:
age(some  missing)
salary(some missing)
city (mixed case: hyderabad, HYDERABAD, Hyderabad)
experience(some duplicates)

perform:
remove duplicates
handle missing values
standardize city names
aplly minMax scaling on age and salary
show final cleaned data

'''

import pandas as pd
import numpy as np

data={
    'age':[20,25,34,20,None],
    'salary':[30000,20000,None,None,10000],
    'city':['BANGALORE','Hyderabad','bangalore','BANGALORE','HYDERABAD'],
    'experience':[1,1,3,6,3]
}

df=pd.DataFrame(data)

'''
remove duplicates
'''
print("Before removing duplicates")
df=df.drop_duplicates()
print("After removing duplicates")
print(df)

'''
handle missing values
'''
print("Before filling missing values")
print(df)
df['age']=df['age'].fillna(df['age'].mean())
df['salary']=df['salary'].fillna(df['salary'].mean())

print("After filling missing values")
print(df)


'''
standardize city names
'''
print("Before standardization")
print(df)

df['city']=df['city'].str.strip().str.lower()
print("After standardization")
print(df)

'''
aplly minMax scaling on age and salary
'''

print("Before MinMaxScaling")
print(df)
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['age','salary']]=scaler.fit_transform(df[['age','salary']])
print("After MinMaxScaling")
print(df)

'''
show final cleaned data
'''
print(df)