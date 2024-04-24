import matplotlib.pyplot as plt
import numpy as np

labels = ['Multinomial NB', 'Bernoulli NB', 'Complement NB', 'Multinomial NB (Boolean)']
sms_scores = np.array([[0.98, 0.95, 0.97], [0.95, 0.93, 0.94], [0.97, 0.96, 0.96], [0.96, 0.94, 0.95]])
email_scores = np.array([[0.94, 0.90, 0.92], [0.90, 0.88, 0.89], [0.93, 0.92, 0.92], [0.91, 0.89, 0.90]])

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, sms_scores[:, 2], width, label='SMS F1-Score')
rects2 = ax.bar(x + width/2, email_scores[:, 2], width, label='Email F1-Score')

ax.set_xlabel('Classifier')
ax.set_ylabel('F1-Score')
ax.set_title('F1-Score by Classifier and Medium')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()