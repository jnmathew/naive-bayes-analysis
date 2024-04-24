# import packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

'''
DATA PROCESSING
'''
# import data
spam_df = pd.read_csv("datasets/sms_data.csv")

# turn ham/spam into 0/1 
spam_df['target'] = spam_df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# create train/test split
x_train, x_test, y_train, y_test = train_test_split(spam_df.Message, spam_df.target, test_size=0.25, random_state=6)

# create vectorizers to transform raw text
cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)

cv_binary = CountVectorizer(binary=True)
x_train_binary = cv_binary.fit_transform(x_train.values)

'''
MODEL TRAINING
'''

# train MNB
model = MultinomialNB()
model.fit(x_train_count, y_train)

# train MNB-B
model_multi_binary = MultinomialNB()
model_multi_binary.fit(x_train_binary, y_train)

# train BNB
model_bernoulli = BernoulliNB()
model_bernoulli.fit(x_train_binary, y_train)

# train CNB
model_complement = ComplementNB()
model_complement.fit(x_train_count, y_train)

'''
MODEL PRE-TESTING 
'''

# pre-test ham
email_ham = ["hey wanna meet up for the game?"]
email_ham_count = cv.transform(email_ham)
email_ham_binary = cv_binary.transform(email_ham)
print("Ham pre-test:\n")
print("MNB:", model.predict(email_ham_count))
print("BNB:", model_bernoulli.predict(email_ham_binary))
print("CNB:", model_complement.predict(email_ham_count))
print("MNB-B:", model_multi_binary.predict(email_ham_binary), "\n")

# pre-test spam
email_spam = ["reward money click free viagra credit card social security"]
email_spam_count = cv.transform(email_spam)
email_spam_binary = cv_binary.transform(email_spam)
print("Spam Pre-test:\n")
print("MNB:", model.predict(email_spam_count))
print("BNB:", model_bernoulli.predict(email_spam_binary)) # strangely, BNB fails pre-test
print("CNB:", model_complement.predict(email_spam_count))
print("MNB-B:", model_multi_binary.predict(email_spam_binary), "\n")

'''
MODEL TESTING AND EVALUATION
'''

x_test_count = cv.transform(x_test)
x_test_binary = cv_binary.transform(x_test)

print("Accuracy scores:")
print("MNB:", model.score(x_test_count, y_test))
print("BNB:", model_bernoulli.score(x_test_binary, y_test))
print("MNB-B", model_multi_binary.score(x_test_binary, y_test))
print("CNB", model_complement.score(x_test_count, y_test), "\n\n")

# reassign y_pred to generate results for different models
y_pred = model_multi_binary.predict(x_test_binary)

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 24, "weight": "bold"}, cmap='Oranges')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: Multinomial NB with Boolean Features (SMS)')
plt.show()