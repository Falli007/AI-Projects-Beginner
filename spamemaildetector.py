import pandas as pd
from sklearn.model_selection import train_test_split # for splitting the dataset
from sklearn.feature_extraction.text import CountVectorizer  # for converting text to numerical data
from sklearn.linear_model import LogisticRegression   # for building the model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score  # for evaluating the model
import seaborn as sns   # for visualizing the results
import matplotlib.pyplot as plt   # for plotting the results

#To load the dataset

df = pd.read_csv('spam.csv', encoding='latin-1')

#To check column names to make sure
df.columns = df.columns.str.strip()  # Removing extra spaces just in case

X = df['text']  # this is the text of the email
y = df['target'] # this is 'ham' or 'spam'


#To convert the text data into numerical data, I will use CountVectorizer
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)


#To split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)


#To build the model, I will use Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

#To make predictions on the test set
y_pred = model.predict(X_test)

#To evaluate the model, I will calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='spam')
recall = recall_score(y_test, y_pred, pos_label='spam')
f1 = f1_score(y_test, y_pred, pos_label='spam')

#Printing the evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

#visualizing the results using seaborn and matplotlib
sns.set(style='whitegrid')
plt.figure(figsize=(10, 6))
sns.barplot(x=['Accuracy', 'Precision', 'Recall', 'F1 Score'], y=[accuracy, precision, recall, f1])
plt.ylim(0, 1)
plt.title('Model Performance Metrics')
plt.xlabel('Metrics')
plt.ylabel('Score')
plt.show()






