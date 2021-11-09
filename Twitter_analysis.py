import json
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

tmp = []
for line in open('./json_file.json','r'):
    tmp.append(json.loads(line))

texts = [tmp[i]['text'] for i in range(len(tmp))]

def get_twitter_text(text):
    stemmer = nltk.stem.SnowballStemmer('english')
    paragraph = stemmer.stem(text)
    vectorizer = CountVectorizer(stop_words='english')
    counts = vectorizer.fit_transform([paragraph])
    counts = pd.Series(counts.toarray()[0], index=vectorizer.get_feature_names())
    return counts



df = []

for i in range(len(texts)):
    df.append(get_twitter_text(texts[i]))
df = [[df[i].index[0],df[0][1]] for i in range(len(df))]

index = []
column = []

for i in range(len(df)):
    index.append(df[i][0])
    column.append(df[i][1])
    
df = pd.DataFrame(column, index=index, columns=['Value'])
l = df.groupby(df.index).sum().sort_values(['Value'], ascending=False).head(20)
plt.figure(figsize=(14, 10))
plt.bar(l.index, l['Value'])
plt.xticks(rotation=45)
plt.show()