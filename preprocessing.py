import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV

col_list = ['overall', 'asin','reviewText', 'summary','vote', 'category', 'title', 'price']
df = pd.read_csv('./combined_data_with_title.csv', usecols = col_list)
df.dropna(inplace=True)
df['Sentiment'] = np.where(df['overall'] > 3, 1, np.where(df['overall']==3, 0, -1))

X_train, X_test, y_train, y_test = train_test_split(df['reviewText'], df['Sentiment'], test_size=0.1, random_state=0)

print('Load %d training examples and %d validation examples. \n' %(X_train.shape[0],X_test.shape[0]))
print('Show a review in the training set : \n', X_train.iloc[10])