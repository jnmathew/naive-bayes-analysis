# import packages
from hard_ham import hard_ham
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# import data
spam_df = pd.read_csv("sms_data.csv")

# inspect data
spam_df.groupby('Category').describe()

# turn ham/spam into 0/1 
spam_df['target'] = spam_df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# create train/test split
x_train, x_test, y_train, y_test = train_test_split(spam_df.Message, spam_df.target, test_size=0.25, random_state=6)

cv = CountVectorizer()
x_train_count = cv.fit_transform(x_train.values)

cv_bernoulli = CountVectorizer(binary=True)
x_train_binary = cv_bernoulli.fit_transform(x_train.values)

#7454 unique words that appear in the 4179 messages
x_train_count

# check matrix as an array
x_train_count.toarray()

# train model
model = MultinomialNB()
model.fit(x_train_count, y_train)

model_multi_binary = MultinomialNB()
model_multi_binary.fit(x_train_binary, y_train)

model_bernoulli = BernoulliNB()
model_bernoulli.fit(x_train_binary, y_train)

model_complement = ComplementNB()
model_complement.fit(x_train_count, y_train)

# pre-test ham
email_ham = ["hey wanna meet up for the game?"]
email_ham_count = cv.transform(email_ham)
email_ham_binary = cv_bernoulli.transform(email_ham)
model.predict(email_ham_count)
model_bernoulli.predict(email_ham_binary)
model_complement.predict(email_ham_count)
print(model_multi_binary.predict(email_ham_binary))

# pre-test spam
email_spam = ["reward money click free"]
email_spam_count = cv.transform(email_spam)
email_spam_binary = cv_bernoulli.transform(email_spam)
print(model.predict(email_spam_count))
print(model_bernoulli.predict(email_spam_binary))
print(model_complement.predict(email_spam_count))
print(model_multi_binary.predict(email_spam_binary))

''' HARD HAM single email check
hard_ham_count = cv.transform(hard_ham)
hard_ham_binary = cv_bernoulli.transform(hard_ham)
hard_ham_tfidf = tfidfv.transform(hard_ham)
model.predict(hard_ham_count)
model_bernoulli.predict(hard_ham_binary)
model_complement.predict(hard_ham_tfidf)
'''

# test model
x_test_count = cv.transform(x_test)
print(model.score(x_test_count, y_test))

x_test_binary = cv_bernoulli.transform(x_test)
print(model_bernoulli.score(x_test_binary, y_test))

print(model_multi_binary.score(x_test_binary, y_test))


print(model_complement.score(x_test_count, y_test))

y_pred = model_multi_binary.predict(x_test_binary)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 24, "weight": "bold"}, cmap='Oranges')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: Multinomial NB with Boolean Features (SMS)')
plt.show()