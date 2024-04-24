# import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# note that vectorizers are not used for email dataset
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

'''
DATA PROCESSING
'''
# import data
spam_df = pd.read_csv("datasets/email_data.csv")

# drop first column
spam_df.drop(columns=spam_df.columns[0], inplace=True)

X = spam_df.iloc[:, :-1] # all columns except last column
y = spam_df.iloc[:, -1] # exclusively last column

# print dataset balance
spam_count = (y == 1).sum()
ham_count = (y == 0).sum()
print(f"Number of spam emails: {spam_count}")
print(f"Number of ham emails: {ham_count}")

# verify 0's and 1's are exhaustive
print(y.value_counts())

'''
MODEL TRAINING
'''

# create train/test split with fixed random_value for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=6)

# create binary versions of training and test sets for BNB & MNB-B
X_train_binary = (X_train > 0).astype(int)
X_test_binary = (X_test > 0).astype(int)

# initialize the Bernoulli NB classifier
bernoulli_model = BernoulliNB()
# train on the binary data
bernoulli_model.fit(X_train_binary, y_train)

# initialize the Multinomial NB classifier
multinomial_model = MultinomialNB()
# train on the original frequency data
multinomial_model.fit(X_train, y_train)

# initialize the Multinomial NB classifier with boolean features
multinomial_binary_model = MultinomialNB()
# train on the binary data
multinomial_binary_model.fit(X_train_binary, y_train)

# initialize the Complement NB classifier
complement_model = ComplementNB()
# train on the original frequency data
complement_model.fit(X_train, y_train)

'''
MODEL TESTING AND EVALUATION
'''

# evaluate Bernoulli NB
bernoulli_accuracy = bernoulli_model.score(X_test_binary, y_test)
bernoulli_report = classification_report(y_test, bernoulli_model.predict(X_test_binary))
print(f'BernoulliNB Accuracy: {bernoulli_accuracy}')
print(f'BernoulliNB Classification Report:\n{bernoulli_report}')

# evaluate Multinomial NB
multinomial_accuracy = multinomial_model.score(X_test, y_test)
multinomial_report = classification_report(y_test, multinomial_model.predict(X_test))
print(f'MultinomialNB Accuracy: {multinomial_accuracy}')
print(f'MultinomialNB Classification Report:\n{multinomial_report}')

# evaluate Multinomial NB with boolean features
multinomial_binary_accuracy = multinomial_binary_model.score(X_test_binary, y_test)
multinomial_binary_report = classification_report(y_test, multinomial_binary_model.predict(X_test_binary))
print(f'MultinomialNB Binary Accuracy: {multinomial_binary_accuracy}')
print(f'MultinomialNB Binary Classification Report:\n{multinomial_binary_report}')

# evaluate Complement NB
complement_accuracy = complement_model.score(X_test, y_test)
complement_report = classification_report(y_test, complement_model.predict(X_test))
print(f'ComplementNB Accuracy: {complement_accuracy}')
print(f'ComplementNB Classification Report:\n{complement_report}')

'''
CONFUSION MATRICES
'''
# print confusion matrix values 
print(confusion_matrix(y_test, multinomial_model.predict(X_test)))
print(confusion_matrix(y_test, bernoulli_model.predict(X_test_binary)))
print(confusion_matrix(y_test, multinomial_binary_model.predict(X_test_binary)))
print(confusion_matrix(y_test, complement_model.predict(X_test)))

# rewrite 2nd param and title to generate confusion matrix for other models
cm = confusion_matrix(y_test, bernoulli_model.predict(X_test_binary))
sns.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 24, "weight": "bold"}, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: Bernoulli NB (Email)')
plt.show()