# import packages
from hard_ham import hard_ham
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# import data
spam_df = pd.read_csv("email_data.csv")

# drop first column
spam_df.drop(columns=spam_df.columns[0], inplace=True)

X = spam_df.iloc[:, :-1]
y = spam_df.iloc[:, -1]

spam_count = (y == 1).sum()
ham_count = (y == 0).sum()

print(f"Number of spam emails: {spam_count}")
print(f"Number of ham emails: {ham_count}")

print(y.value_counts())

# create train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=6)

# Convert training and test sets to binary format
X_train_binary = (X_train > 0).astype(int)
X_test_binary = (X_test > 0).astype(int)

# Initialize the Bernoulli Naive Bayes classifier
bernoulli_model = BernoulliNB()
# Train on the binary data
bernoulli_model.fit(X_train_binary, y_train)

# Initialize the Multinomial Naive Bayes classifier
multinomial_model = MultinomialNB()
# Train on the original frequency data
multinomial_model.fit(X_train, y_train)

# Initialize the Multinomial Naive Bayes classifier with Binary Features
multinomial_binary_model = MultinomialNB()
# Train on the original frequency data
multinomial_binary_model.fit(X_train_binary, y_train)

# Initialize the Complement Naive Bayes classifier
complement_model = ComplementNB()
# Train on the original frequency data
complement_model.fit(X_train, y_train)

# Evaluate Bernoulli Naive Bayes
bernoulli_accuracy = bernoulli_model.score(X_test_binary, y_test)
bernoulli_report = classification_report(y_test, bernoulli_model.predict(X_test_binary))
print(f'BernoulliNB Accuracy: {bernoulli_accuracy}')
print(f'BernoulliNB Classification Report:\n{bernoulli_report}')

# Evaluate Multinomial Naive Bayes
multinomial_accuracy = multinomial_model.score(X_test, y_test)
multinomial_report = classification_report(y_test, multinomial_model.predict(X_test))
print(f'MultinomialNB Accuracy: {multinomial_accuracy}')
print(f'MultinomialNB Classification Report:\n{multinomial_report}')

# Evaluate Multinomial Binary Naive Bayes
multinomial_binary_accuracy = multinomial_binary_model.score(X_test_binary, y_test)
multinomial_binary_report = classification_report(y_test, multinomial_binary_model.predict(X_test_binary))
print(f'MultinomialNB Binary Accuracy: {multinomial_binary_accuracy}')
print(f'MultinomialNB Binary Classification Report:\n{multinomial_binary_report}')

# Evaluate Complement Naive Bayes
complement_accuracy = complement_model.score(X_test, y_test)
complement_report = classification_report(y_test, complement_model.predict(X_test))
print(f'ComplementNB Accuracy: {complement_accuracy}')
print(f'ComplementNB Classification Report:\n{complement_report}')

'''
tfidfv = TfidfVectorizer()
x_train_tfidf = tfidfv.fit_transform(x_train.values)

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
model_complement.fit(x_train_tfidf, y_train)


HARD HAM single email check
hard_ham_count = cv.transform(hard_ham)
hard_ham_binary = cv_bernoulli.transform(hard_ham)
hard_ham_tfidf = tfidfv.transform(hard_ham)
model.predict(hard_ham_count)
model_bernoulli.predict(hard_ham_binary)
model_complement.predict(hard_ham_tfidf)


# test model
x_test_count = cv.transform(x_test)
print(model.score(x_test_count, y_test))

x_test_binary = cv_bernoulli.transform(x_test)
print(model_bernoulli.score(x_test_binary, y_test))

print(model_multi_binary.score(x_test_binary, y_test))

x_test_tfidf = tfidfv.transform(x_test)

y_pred = model_complement.predict(x_test_binary)

print(model_complement.score(x_test_tfidf, y_test))

cv = CountVectorizer()
cv_bernoulli = CountVectorizer(binary=True)

# pre-test ham
email_ham = ["hey wanna meet up for the game?"]
email_ham_count = cv.transform(email_ham)
email_ham_binary = cv_bernoulli.transform(email_ham)
print(multinomial_model.predict(email_ham_count))
print(bernoulli_model.predict(email_ham_binary))
print(complement_model.predict(email_ham_count))
print(multinomial_binary_model.predict(email_ham_binary))

# pre-test spam
email_spam = ["reward money click free"]
email_spam_count = cv.transform(email_spam)
email_spam_binary = cv_bernoulli.transform(email_spam)
print(multinomial_model.predict(email_spam_count))
print(bernoulli_model.predict(email_spam_binary))
print(complement_model.predict(email_spam_count))
print(multinomial_binary_model.predict(email_spam_binary))

'''

print(confusion_matrix(y_test, multinomial_model.predict(X_test)))
print(confusion_matrix(y_test, bernoulli_model.predict(X_test_binary)))
print(confusion_matrix(y_test, multinomial_binary_model.predict(X_test_binary)))
print(confusion_matrix(y_test, complement_model.predict(X_test)))


'''
print(classification_report(y_test, multinomial_model.predict(X_test)))
'''

cm = confusion_matrix(y_test, bernoulli_model.predict(X_test_binary))
sns.heatmap(cm, annot=True, fmt='d', annot_kws={"size": 24, "weight": "bold"}, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix: Bernoulli NB (Email)')
plt.show()