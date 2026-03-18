'''
Task for Today:

You are given a csv file containing:
name,maths,science,english,dept

requirement:
read csv file,
add new column total,
add a column average
find: top 3 studnets based on total
dept wise average marks
students scoring above 75 in all subjects
handle missing values(if any)
'''
import pandas as pd
df=pd.read_csv('students_data.csv')
print(df)
df['total']=df['maths']+df['science']+df['english']
print(df)
df['average']=df['total']/3
df.nlargest(3,'total')
df.groupby('dept')['total'].mean()
filtered_rows=df[(df['maths']>75) & (df['science']>75) & (df['english']>75)]
print(filtered_rows)
df.isnull()
df['maths']=df['maths'].fillna(df['maths'].mean())
df['english']=df['english'].fillna(df['english'].mean())
print(df)


