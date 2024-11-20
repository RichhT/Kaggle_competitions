import pandas as pd
from sklearn import svm 
from sklearn import metrics
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/train.csv')

test_df=pd.read_csv('data/test.csv')

# save the ids separately because they're removed upon reindexing
test_df_id = test_df['id']


## PRE-PROCESS TRAINING DATA

# Drop these columns because they have >80% missing values
df = df.drop(columns=['Academic Pressure', 'CGPA', 'Study Satisfaction'])
# Check levels of missing data now
missing_percentage = df.isnull().mean() * 100
print(missing_percentage)
# Drop the remaining missing data
df = df.dropna()
# Check all data is now complete
missing_percentage = df.isnull().mean() * 100
print(missing_percentage)
df = pd.get_dummies(df, columns=['Gender', 'City', 'Working Professional or Student', 'Profession', 'Sleep Duration', 'Dietary Habits','Degree', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness'], drop_first='True')


X_train, X_test, y_train, y_test = train_test_split(df.drop(columns=['id', 'Name', 'Depression']), df['Depression'], test_size=0.3, random_state=0)


## PRE-PROCESS TEST DATA
test_df = test_df.drop(columns=['Academic Pressure', 'CGPA', 'Study Satisfaction'], errors='ignore')
test_df = pd.get_dummies(test_df, columns=['Gender', 'City', 'Working Professional or Student', 'Profession', 'Sleep Duration', 'Dietary Habits','Degree', 'Have you ever had suicidal thoughts ?', 'Family History of Mental Illness'], drop_first='True')
test_df = test_df.reindex(columns=X_train.columns, fill_value=0)
test_df = test_df.dropna()

# clf = svm.SVC(kernel='poly')

# clf.fit(X_train, y_train)

# test_df['Depression'] = clf.predict(test_df)

# test_df['id'] = test_df_id

# submission = test_df[['id', 'Depression']]

# submission.to_csv('submission.csv', index=False)