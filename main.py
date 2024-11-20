import pandas as pd
from sklearn import svm 
from sklearn import metrics
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/train.csv')

#print(df.head())

#print(df.columns)

#x_df = df.drop(columns=['id', 'Name', 'Depression'])

#print(df.head())
#print(x_df.head())

df = df.drop(columns=['Academic Pressure', 'CGPA', 'Study Satisfaction'])

missing_percentage = df.isnull().mean() * 100
print(missing_percentage)

df = df.dropna()
missing_percentage = df.isnull().mean() * 100
print(missing_percentage)


# print(df.dtypes)

# df = pd.get_dummies(df, columns=['Gender', 'City', 'Working Professional or Student', 'Profession', 'Sleep Duration', 'Dietary Habits','Degree', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness'], drop_first='True')

# print(df.head())

# X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['id', 'Name', 'Depression']), df['Depression'], test_size=0.3, random_state=0)
# clf = svm.SVC(kernel='poly')

# clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)