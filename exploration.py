import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

col_list = ['overall', 'asin','reviewText', 'summary','vote', 'category', 'title', 'price']
df = pd.read_csv('./combined_data_with_title.csv', usecols = col_list)

print("\nTotal number of reviews: ",len(df))
print("\nTotal number of categories: ", len(list(set(df['category']))))
print("\nTotal number of unique products: ", len(list(set(df['title']))))

print("\nPercentage of reviews with neutral sentiment : {:.2f}%"\
      .format(df[df['overall']==3]["overall"].count()/len(df)*100))
print("\nPercentage of reviews with positive sentiment : {:.2f}%"\
      .format(df[df['overall']>3]["overall"].count()/len(df)*100))
print("\nPercentage of reviews with negative sentiment : {:.2f}%"\
      .format(df[df['overall']<3]["overall"].count()/len(df)*100))

# Plot distribution of rating
plt.figure(figsize=(12,8))
# sns.countplot(df['Rating'])
df['overall'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of Rating')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()


# Plot number of reviews for top 20 brands
brands = df["category"].value_counts()
# brands.count()
plt.figure(figsize=(12,8))
brands[:20].plot(kind='bar')
plt.title("Number of Reviews for Top 20 category")
plt.show()


# Plot number of reviews for top 50 products
products = df["title"].value_counts()
plt.figure(figsize=(12,8))
products[:50].plot(kind='bar')
plt.title("Number of Reviews for Top 50 Products")
plt.show()

# Plot distribution of review length
review_length = df["reviewText"].dropna().map(lambda x: len(x))
plt.figure(figsize=(12,8))
review_length.loc[review_length < 1500].hist()
plt.title("Distribution of Review Length")
plt.xlabel('Review length (Number of character)')
plt.ylabel('Count')
plt.show()