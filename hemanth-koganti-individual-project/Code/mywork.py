# converting (genre) json column to normal string column
# Replacing null values with '{}'
df_cleaned['genres'] = df_cleaned['genres'].replace(np.nan,'{}',regex = True)
# Converting Strings to Dictionaries as it have multiple genres in json format
df_cleaned['genres'] = pd.DataFrame(df_cleaned['genres'].apply(eval))
# dividing all genres in a cell into separate cols/series, concatenating it to main df & then dropping the original "genres" column from df
df_cleaned = pd.concat([df_cleaned.drop(['genres'], axis=1), df_cleaned['genres'].apply(pd.Series)], axis=1)
# Removing all columns except the major genre type for each movie
df_cleaned.drop(df_cleaned.iloc[:, 15:], inplace = True, axis = 1)
# creating separate series for "id" & "name" and concatenating it to main df
df_cleaned = pd.concat([df_cleaned.drop([0], axis=1), df_cleaned[0].apply(pd.Series)], axis=1)
df_cleaned.drop(df_cleaned.iloc[:, 14:16], inplace = True, axis = 1)     # dropping extraneous cols
df_cleaned.rename(columns = {'name' : 'Genre'}, inplace = True)   # renaming col
df_cleaned = df_cleaned[~df_cleaned['Genre'].isnull()] # removing null containing rows


# converting (production_companies) json column to normal string column
# Replacing null values with '{}'
df_cleaned['production_companies'] = df_cleaned['production_companies'].replace(np.nan,'{}',regex = True)
# Converting Strings to Dictionaries as it have multiple production companies in json format
df_cleaned['production_companies'] = pd.DataFrame(df_cleaned['production_companies'].apply(eval))
# Dividing all production companies into separate cols, concatenating these to the main df and dropping the original 'production companies' col
df_cleaned = pd.concat([df_cleaned.drop(['production_companies'], axis=1), df_cleaned['production_companies'].apply(pd.Series)], axis=1)
# Removing all production companies cols except major production company for each movie.
df_cleaned.drop(df_cleaned.iloc[:, 14:], inplace = True, axis = 1)
# creating separate series for "name" & "id" and concatenating it to main df
df_cleaned = pd.concat([df_cleaned.drop([0], axis=1), df_cleaned[0].apply(pd.Series)], axis=1)
# dropping unnecessary cols
df_cleaned.drop(df_cleaned.iloc[:, 13:15], inplace = True, axis = 1)
# renaming newly created col
df_cleaned.rename(columns = {'name' : 'Production_Company'}, inplace = True)
df_cleaned = df_cleaned[~df_cleaned['Production_Company'].isnull()]
len(df_cleaned.Production_Company.unique())
# =================================================================
# EDA
# =================================================================
#
# summary = merged_inner.describe()
#
# dependent variable
ax = sns.countplot(merged_inner["New_status"])
ax.set(xlabel ='Labels', ylabel ='Frequency')
plt.title("Target Variable",fontsize=20)
plt.show()
#
#
# # status column
max_profit = merged_inner["status"].max()    # 653 times
max_profit_movie = merged_inner.loc[merged_inner['status'] == max_profit]   # The way of the dragon (director Bruce Lee)
#
ax = sns.distplot(merged_inner["status"], bins=500, kde=False)
# control x and y limits
ax.set(xlabel ='Ratio', ylabel ='Frequency')
plt.title("Revenue/Budget Ratio",fontsize=20)
plt.ylim(0, 700)
plt.xlim(-1, 50)
plt.show()
#
# #1.  independent variable Year
merged_inner["startYear"].min()     # 1921
merged_inner["startYear"].max()     # 2017
#
decades = []
for each in merged_inner["startYear"]:
    decade = int(np.floor(each / 10) * 10)
    decades.append(decade)
#
ax = sns.countplot(decades)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set(xlabel ='Decades', ylabel ='Frequency')
plt.title("Movie Count by Decades",fontsize=20)
plt.show()
#
#
# # 2. independent variable Runtime
ax = sns.distplot(merged_inner['runtime'], kde=False, rug=False);
ax.set(xlabel ='Duration in Minutes', ylabel ='Frequency')
plt.title("Movie Runtime",fontsize=20)
plt.show()
#
#
# # 3. independent variable Vote Average
ax = sns.distplot(merged_inner['averageRating'], kde=False, rug=False);
ax.set(xlabel ='Movie Rating', ylabel ='Frequency')
plt.title("Movie Rating",fontsize=20)
plt.show()
#
max_rating_movie = merged_inner.loc[merged_inner['averageRating'] == 9.3]   # The Shawshank Redemption
#
# # 4. independent variable Production Company
plt.figure(figsize=(20,12))
sns.countplot(merged_inner['Production_Company'], order=merged_inner.Production_Company.value_counts().iloc[:10].index)
plt.title("Top 10 Production companies",fontsize=20)
plt.show()
#
# # 5. independent variable release month
plt.figure(figsize=(20,12))
sns.countplot(merged_inner['release_month'].sort_values())
plt.title("Movies by Release month",fontsize=20)
plt.show()
#
# # 6. independent variable Vote Count
ax = sns.distplot(merged_inner["numVotes"], kde=False, rug=False)
ax.set(xlabel ='Vote Distribution', ylabel ='Frequency')
plt.title("Vote Count",fontsize=20)
plt.show()
#
max_vote_movie = merged_inner.loc[merged_inner['numVotes'] == 2162821]   # The Shawshank Redemption
#
# # 7. Number of movies per Genre
a = merged_inner["Genre"].unique()    # 18
ax = sns.countplot(merged_inner["Genre"])
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set(xlabel ='Genre', ylabel ='Frequency')
plt.title("Movie Count by Genre",fontsize=20)
plt.show()
#
#
# # 8. Movie Budget
ax = sns.distplot(merged_inner["budget"], kde=False, rug=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
ax.set(xlabel ='Budget', ylabel ='Frequency')
xlabels = ['{:,.2f}'.format(x) + 'M' for x in ax.get_xticks()/1000000]
ax.set_xticklabels(xlabels)
plt.title("Movie Budget",fontsize=20)
plt.show()
#
#
# # Correlation heatmap bw numerical cols
merged_inner["popularity"] = merged_inner["popularity"].astype(float).fillna(0.0)
num_cols = merged_inner[['budget', 'startYear', 'revenue', 'runtime', 'popularity', 'averageRating', 'numVotes','status']]
# # removing redundant upper half of heat map
mask = np.zeros(num_cols.corr().shape, dtype=bool)
mask[np.triu_indices(len(mask))] = True
sns.heatmap(num_cols.corr(), annot=True, vmin = -1, vmax = 1, center = 0, cmap = 'coolwarm', mask = mask)
plt.show()