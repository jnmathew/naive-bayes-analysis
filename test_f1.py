import pandas as pd
import matplotlib.pyplot as plt

# Define the data
data = {
    'Classifier': ['MNB', 'MNB', 'BNB', 'BNB', 'MNB-B', 'MNB-B', 'CNB', 'CNB',
                   'MNB', 'MNB', 'BNB', 'BNB', 'MNB-B', 'MNB-B', 'CNB', 'CNB'],
    'Class': ['Ham', 'Spam', 'Ham', 'Spam', 'Ham', 'Spam', 'Ham', 'Spam',
              'Ham', 'Spam', 'Ham', 'Spam', 'Ham', 'Spam', 'Ham', 'Spam'],
    'F1-Score': [1.00, 0.96, 0.99, 0.92, 1.00, 0.96, 0.99, 0.92,
                 0.96, 0.91, 0.90, 0.77, 0.95, 0.89, 0.96, 0.91],
    'Medium': ['SMS', 'SMS', 'SMS', 'SMS', 'SMS', 'SMS', 'SMS', 'SMS',
               'Email', 'Email', 'Email', 'Email', 'Email', 'Email', 'Email', 'Email']
}

# Create a DataFrame
df = pd.DataFrame(data)

# Pivoting the data for easier plotting
pivot_df = df.pivot_table(index=['Classifier', 'Medium'], columns='Class', values='F1-Score').reset_index()

# Setting the positions and width for the bars
positions = range(len(pivot_df['Classifier'].unique()))
bar_width = 0.15

fig, ax = plt.subplots(figsize=(12, 6))

# Plotting bars for each class
rects1 = ax.bar(positions, pivot_df[pivot_df['Medium'] == 'SMS']['Ham'], bar_width, label='SMS - Ham', color="#7F2704")
rects2 = ax.bar([p + bar_width for p in positions], pivot_df[pivot_df['Medium'] == 'SMS']['Spam'], bar_width, label='SMS - Spam', color="#FEE5CC")
rects3 = ax.bar([p + 2*bar_width for p in positions], pivot_df[pivot_df['Medium'] == 'Email']['Ham'], bar_width, label='Email - Ham', color="#08306B")
rects4 = ax.bar([p + 3*bar_width for p in positions], pivot_df[pivot_df['Medium'] == 'Email']['Spam'], bar_width, label='Email - Spam', color="#89BEDC")

# Adding labels, title, and custom x-axis tick labels
ax.set_xlabel('Classifier')
ax.set_ylabel('F1-Score')
ax.set_title('F1-Scores by Classifier, Class, and Medium')
ax.set_xticks([p + 1.5*bar_width for p in positions])
ax.set_xticklabels(pivot_df['Classifier'].unique())
ax.legend()

# Adding a legend and showing the plot
plt.legend()
plt.tight_layout()
plt.show()