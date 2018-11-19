# -*- coding: utf-8 -*-
"""
Created on Wed May  2 15:02:00 2018

@author: Gaurav Bobade
"""
#import libraries

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

df= pd.read_csv('D:\\Spring18\\DataScienceforPython\\Project\\TEDTalksDataset\\ted_main.csv')

df.columns
# Exploration of the fields in the dataset
df['views'].describe()
df['comments'].describe()
df['ratings'].describe()
df['duration'].describe()
df['languages'].describe()

#Conversion of unix timestamps to human readable timestamps
df['film_date'] = df['film_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))

df['published_date'] = df['published_date'].apply(lambda x: datetime.datetime.fromtimestamp( int(x)).strftime('%d-%m-%Y'))
df.head(10)

# gives the number of rows the dataset contains
len(df)

#Most Popular, Least Popular TED Talks and highest comments for a TED Talks

pop_talks = df[['title', 'main_speaker', 'views','languages']].sort_values('views', ascending=False)[:15]
pop_talks

least_pop_talks = df[['title', 'main_speaker', 'views', 'languages']].sort_values('views', ascending=True)[:15]
least_pop_talks

most_comments = df[['title', 'main_speaker','views','comments']].sort_values('comments', ascending =False)[:10]
most_comments

#Bar Plot for 15 most viewed TED Talks:
pop_talks['abbr'] = pop_talks['main_speaker'].apply(lambda x: x[:3])
sns.set_style("whitegrid")
plt.figure(figsize=(15,6))
sns.barplot(x='abbr', y='views', data=pop_talks)

#Bar plot of Speaker and number of languages the TED Talk has been translated
pop_talks['Speaker'] = pop_talks['main_speaker'].apply(lambda x: x[:3]) 

sns.set_style("whitegrid")
plt.figure(figsize=(15,5))
sns.barplot(x='Speaker', y='languages', data=pop_talks)


#Lets check TED Talks with longest duration 
#Then compare longest duration speakers with most popular speakers
# Note: Duration is in seconds
highest_dur = df[['title', 'main_speaker', 'views', 'film_date','speaker_occupation','duration']].sort_values('duration', ascending=False)[:15]
highest_dur

highest_dur['Speaker'] = highest_dur['main_speaker'] .apply(lambda x: x[:5])
sns.set_style("whitegrid")
plt.figure(figsize=(25,5))
sns.barplot(x='Speaker', y='duration', data=highest_dur)

#Statistic distribution using plots
sns.distplot(df['duration'])
df['duration'].describe()
#checking distribution of values
sns.distplot(df['languages'])

sns.distplot(df['views'])

sns.distplot(df['languages'], bins=20, hist = True, rug = False)

df['comments'].describe()
sns.distplot(df['comments'])
sns.distplot(df[df['comments'] < 500]['comments'])

# correlation:
sns.jointplot(x='views', y='comments', data=df)
df[['views', 'comments']].corr()

#Content is king at TED.
sns.jointplot(x='views', y='duration', data=df)
df[['views', 'duration']].corr()

sns.jointplot(x='languages', y='views', data=df)

#Militant atheism had less views than Ken Robinson but had most comments
df[['title', 'main_speaker','views', 'comments']].sort_values('comments', ascending=False).head(10)

#TED Talks by month 

month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
day_order = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

df['month'] = df['film_date'].apply(lambda x: month_order[int(x.split('-')[1]) - 1])

month_df = pd.DataFrame(df['month'].value_counts()).reset_index()
month_df.columns = ['month', 'talks']

#February is clearly the most popular month for TED talk events whereas August and January are the least popular. 
sns.barplot(x='month', y='talks', data=month_df, order=month_order)
#TED Talks over the years
df['year'] = df['film_date'].apply(lambda x: x.split('-')[2])
year_df = pd.DataFrame(df['year'].value_counts().reset_index())
year_df.columns = ['year', 'talks']

plt.figure(figsize=(20,5))
sns.pointplot(x='year', y='talks', data=year_df)

#let us construct a heat map that shows us the number of talks by month and year. This will give us a good summary of the distribution of talks.
months = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
#heat map code
hmap_df = df.copy()
hmap_df['film_date'] = hmap_df['film_date'].apply(lambda x: month_order[int(x.split('-')[1]) - 1] + " " + str(x.split('-')[2]))
hmap_df = pd.pivot_table(hmap_df[['film_date', 'title']], index='film_date', aggfunc='count').reset_index()
hmap_df['month'] = hmap_df['film_date'].apply(lambda x: months[x.split()[0]])
hmap_df['year'] = hmap_df['film_date'].apply(lambda x: x.split()[1])
hmap_df = hmap_df.sort_values(['year', 'month'])
hmap_df = hmap_df[['month', 'year', 'title']]
hmap_df = hmap_df.pivot('month', 'year', 'title')
hmap_df = hmap_df.fillna(0)
#plotting heatmap
f, ax = plt.subplots(figsize=(15, 8))
sns.heatmap(hmap_df, annot=True, linewidths=.5, ax=ax, fmt='n', yticklabels=month_order)

#Speakers and thier occupations.Their appearance on TED Talks 
speaker_df = df.groupby('main_speaker').count().reset_index()[['main_speaker', 'comments']]
speaker_df
#Speaker who appeared for the most number of times
speaker_df.columns = ['main_speaker', 'appearances']
speaker_df = speaker_df.sort_values('appearances', ascending=False)
speaker_df.head(10)

occupation_df = df.groupby('speaker_occupation').count().reset_index()[['speaker_occupation', 'comments']]
occupation_df.columns = ['occupation', 'appearances']
occupation_df = occupation_df.sort_values('appearances', ascending=False)
# Occupation wise  bar plot
occupation_df.head(10)
plt.figure(figsize=(20,5))
sns.barplot(x='occupation', y='appearances', data=occupation_df.head(10))
plt.show()

#Box Plot
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=(15, 8))
sns.boxplot(x='speaker_occupation', y='views', data=df[df['speaker_occupation'].isin(occupation_df.head(10)['occupation'])], palette="muted", ax =ax)
ax.set_ylim([0, 0.4e7])
plt.show()

#Number of speakers for particular TED Talk
df['num_speaker'].value_counts()

#KMeans Clustering
#Cluster 1 "View and Comments" 
df_combined = df[['views', 'comments']]
df_combined 

kmeans = KMeans(n_clusters=3)
kmeans.fit(df_combined)
y_pred = kmeans.predict(df_combined)

centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(10,10))

plt.scatter(df_combined['views'], df_combined['comments'], c=y_pred, alpha=0.5)
plt.show()

#Cluster 2 "Views and Duration"

df_combined1 = df[['views', 'duration']]
df_combined1

kmeans = KMeans(n_clusters=3)
kmeans.fit(df_combined1)

y_pred = kmeans.predict(df_combined1)
centroids = kmeans.cluster_centers_

fig = plt.figure(figsize=(10,10))

plt.scatter(df_combined1['views'], df_combined1['duration'], c=y_pred, alpha=0.5)
plt.show()

#Themes in TED Talks

import ast
df['tags'] = df['tags'].apply(lambda x: ast.literal_eval(x))
s = df.apply(lambda x: pd.Series(x['tags']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'theme'
theme_df = df.drop('tags', axis=1).join(s)
theme_df.head()

len(theme_df['theme'].value_counts())

#Most popular themes and barplot
pop_themes = pd.DataFrame(theme_df['theme'].value_counts()).reset_index()
pop_themes.columns = ['theme', 'talks']
pop_themes.head(10)

plt.figure(figsize=(15,5))
sns.barplot(x='theme', y='talks', data=pop_themes.head(10))
plt.show()
