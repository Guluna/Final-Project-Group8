import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score



# reading csv file data
movie_data_orig = pd.read_csv("movies_metadata.csv")
# print(movie_data_orig)     # [45466 rows x 24 columns]

# removing 12 irrelevant columns
df_cleaned = movie_data_orig.drop(["adult", "belongs_to_collection", "homepage", "original_language",
                                   "original_title", "overview", "poster_path", "production_countries",
                                   "spoken_languages", "status", "tagline", "video" ], axis=1)

# print(df_cleaned.columns)   #  'budget', 'genres', 'id', 'imdb_id' 'popularity', 'production_companies', 'release_date', 'revenue', 'runtime', 'title', 'vote_average', 'vote_count']
# df_cleaned.dtypes       # release_date is of object (i.e. string data type) instead of datetime

#::-------------------------------------------------------
# Problem1 = Python converting movies released in before 1969 to 2059 e.g.
#::-

# converting release_date column from object to datetime format
print(df_cleaned["release_date"])
df_cleaned["release_date"] = pd.to_datetime(df_cleaned["release_date"], format="%m/%d/%y", errors="coerce")   #  errors=‘coerce’, then invalid parsing will be set as NaN
print(len(df_cleaned.loc[df_cleaned["release_date"].isnull() == True]))      # 168 so we can remove these rows from our df
df_cleaned.dtypes        # release_date is now datetime64[ns]
df_cleaned["release_date"].min()    #   Timestamp('1969-01-01 00:00:00')
print(df_cleaned["release_date"])


a = df_cleaned.loc[df_cleaned["title"] == "North by Northwest"]   # release_date should be 1959 but py conv it to 2059. Why ????

#::-------------------------------------------------------
# Problem1
#::-


# budget column contains alpha-numeric characters, so need to fix it
df_cleaned['budget'] = df_cleaned['budget'].str.extract('(\d+)', expand=False)   # removing all non-numeric values from budget column
# changing budget column from object to float
df_cleaned["budget"] = df_cleaned["budget"].astype(float).fillna(0.0)
df_cleaned = df_cleaned.loc[(df_cleaned['budget'] > 100000) & (df_cleaned['revenue'] > 1000)]   # subsetting df to only movies with budget greater than $100,000 & revenue greater than $1000
# df_cleaned = df_cleaned.loc[]      # subsetting df to only movies with

# creating our target/label column showing status i.e success/flop movie.
df_cleaned["status"] = df_cleaned["revenue"]/df_cleaned["budget"]
# Our criteria for success is any value greater than 1 else flop
df_cleaned["New_status"] = np.nan      # creating a new empty target column called New_Status
df_cleaned["New_status"] = df_cleaned["New_status"].mask( df_cleaned["status"] > 1, 1)
df_cleaned["New_status"] = df_cleaned["New_status"].mask( df_cleaned["status"] <= 1, 0)
df_cleaned["New_status"] = df_cleaned["New_status"].astype("category")      # converting from float to categorical datatye

# there are many entries where the number of people who voted for a movie are 1, 2 , 3 etc. They need to be removed otherwise it will create bias
df_cleaned = df_cleaned.loc[(df_cleaned['vote_count'] > 100) & df_cleaned["vote_average"] > 0]      # subsetting df to only movies where atleast 100 people voted for a movie & vote_average > 0

# rearranging columns of dataframe
cols = df_cleaned.columns.tolist()
# Setting Genre as last col for easier manipulation
cols = ['budget', 'id', 'imdb_id', 'popularity', 'production_companies', 'release_date', 'revenue', 'runtime', 'title', 'vote_average', 'vote_count', 'status', 'New_status', 'genres']
df_cleaned = df_cleaned[cols]

# converting (genre) json column to normal string column
# Replacing null values with '{}'
df_cleaned['genres'] = df_cleaned['genres'].replace(np.nan,'{}',regex = True)
df_cleaned['genres'] = pd.DataFrame(df_cleaned['genres'].apply(eval))
# dividing all genres in a cell into separate cols/series, concating it to main df & then dropping the original "genres" column from df
df_cleaned = pd.concat([df_cleaned.drop(['genres'], axis=1), df_cleaned['genres'].apply(pd.Series)], axis=1)
# Removing all columns except the major genre type for each movie
df_cleaned.drop(df_cleaned.iloc[:, 14:], inplace = True, axis = 1)
# creating separate series for "id" & "name" and concating it to main df
df_cleaned = pd.concat([df_cleaned.drop([0], axis=1), df_cleaned[0].apply(pd.Series)], axis=1)
df_cleaned.drop(df_cleaned.iloc[:, 13:15], inplace = True, axis = 1)     # dropping extraneous cols
df_cleaned.rename(columns = {'name' : 'Genre'}, inplace = True)   # renaming col
df_cleaned = df_cleaned[~df_cleaned['Genre'].isnull()] # removing null containing rows


# rearranging columns of dataframe
cols = df_cleaned.columns.tolist()
# Setting Production_companies as last col for easier manipulation
cols = ['budget', 'imdb_id', 'popularity', 'release_date', 'revenue', 'runtime', 'title', 'vote_average', 'vote_count', 'status', 'New_status', 'Genre', 'production_companies']
df_cleaned = df_cleaned[cols]

# converting (production_companies) json column to normal string column
# Replacing null values with '{}'
df_cleaned['production_companies'] = df_cleaned['production_companies'].replace(np.nan,'{}',regex = True)
# Converting Strings to Dictionaries as it have multiple production companies in json format
df_cleaned['production_companies'] = pd.DataFrame(df_cleaned['production_companies'].apply(eval))
# Dividing all production companies into separate cols, concatenating these to the main df and dropping the original 'production companies' col
df_cleaned = pd.concat([df_cleaned.drop(['production_companies'], axis=1), df_cleaned['production_companies'].apply(pd.Series)], axis=1)
# Removing all production companies cols except major production company
df_cleaned.drop(df_cleaned.iloc[:, 13:], inplace = True, axis = 1)
# creating separate series for "name" & "id" and concating it to main df
df_cleaned = pd.concat([df_cleaned.drop([0], axis=1), df_cleaned[0].apply(pd.Series)], axis=1)
# dropping unnecessary cols
df_cleaned.drop(df_cleaned.iloc[:, 12:14], inplace = True, axis = 1)
# renaming newly created col
df_cleaned.rename(columns = {'name' : 'Production Company'}, inplace = True)
df_cleaned = df_cleaned[~df_cleaned['Production Company'].isnull()]

# Adding Director col using imdb files
dir_id_imdb = pd.read_csv('title_crew.tsv', sep='\t')
merged_inner = pd.merge(left=df_cleaned,right=dir_id_imdb, left_on='imdb_id', right_on='tconst')
dir_name_imdb = pd.read_csv('name_basics.tsv', sep='\t')
merged_inner = pd.merge(left=merged_inner,right=dir_name_imdb, left_on='directors', right_on='nconst')
merged_inner = merged_inner.drop(["tconst", "directors", "nconst"], axis=1)     # removing irrelevant cols
merged_inner.rename(columns = {'primaryName' : 'Director'}, inplace = True)


# Adding Avg_ratings & Total votes cols using imdb files
ratings_imdb = pd.read_csv('title_ratings.tsv', sep='\t')
merged_inner = pd.merge(left=merged_inner,right=ratings_imdb, left_on='imdb_id', right_on='tconst')
merged_inner = merged_inner.drop(["tconst", "vote_average", "vote_count"], axis=1)     # removing old vote_avg/count cols

# Adding Movie release year column from imdb file
releaseYr_imdb = pd.read_csv('title_year.tsv', sep='\t')
merged_inner = pd.merge(left=merged_inner,right=releaseYr_imdb, left_on='imdb_id', right_on='tconst')
merged_inner = merged_inner.drop(["tconst"], axis=1)
cols = merged_inner.columns.tolist()
# Setting StartYear col beside release_date col
cols = ['budget', 'imdb_id', 'popularity', 'release_date', 'startYear', 'revenue', 'runtime', 'title', 'status', 'New_status', 'Genre', 'Production Company', 'Director', 'averageRating', 'numVotes']
merged_inner = merged_inner[cols]

len(merged_inner.Director.unique())     # 1173

len(merged_inner)

# Removing Duplicates
merged_inner.drop_duplicates(inplace = True)

# =================================================================
# EDA
# =================================================================


# #
# plt.figure(figsize=(20,12))
# sns.countplot(df_cleaned['vote_average'].sort_values())
# plt.title("Rating Count",fontsize=20)
# plt.show()
#
# # Number of movies per Genre
# plt.figure(figsize=(20,12))
# sns.countplot(df_cleaned['Genre'])
# plt.title("Genre Count",fontsize=20)
# plt.show()
#
# # Correlation heatmap
# df_c = df_cleaned[['budget','revenue','runtime','vote_average','vote_count']]
# f,ax = plt.subplots(figsize=(10, 5))
# sns.heatmap(df_c.corr(), annot=True)
# plt.show()

# Pair Plot
# df_x = df_cleaned[['budget','revenue','runtime','vote_average','vote_count','New_status']]
# sns.set(style = 'ticks')
# sns.pairplot(df_x, hue = 'New_status')
# plt.show()



# =================================================================
# Modeling
# =================================================================
#
#Decision Tree Gini
# split the dataset into input and target variables
# X = df_cleaned.loc[:,['runtime','vote_average','Genre','release_month','Production Company']]
# y = df_cleaned.loc[:,['New_status']]
#
# # encloding the class with sklearn's LabelEncoder
# le = LabelEncoder()
#
# # fit and transform the class
# y = le.fit_transform(y)
# X = pd.get_dummies(X)
#
# # split the dataset into train and test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
#
# # perform training with giniIndex.
# # creating the classifier object
# clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
#
# # performing training
# clf_gini.fit(X_train, y_train)
#
# # predicton on test using gini
# y_pred_gini = clf_gini.predict(X_test)
#
# print("Classification Report: ")
# print(classification_report(y_test,y_pred_gini))
# print("Accuracy : ", accuracy_score(y_test, y_pred_gini) * 100)
#
# #Decision Tree Entropy
# # perform training with Entropy.
# # creating the classifier object
# clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
#
# # performing training
# clf_entropy.fit(X_train, y_train)
#
# # predicton on test using gini
# y_pred_entropy = clf_entropy.predict(X_test)
#
# print("Classification Report: ")
# print(classification_report(y_test,y_pred_entropy))
# print("Accuracy : ", accuracy_score(y_test, y_pred_entropy) * 100)
#
# #Applying SVM Classification
# # perform training
# # creating the classifier object
# clf = SVC(kernel="linear")
#
# # performing training
# clf.fit(X_train, y_train)
#
# # predicton on test
# y_pred_svm = clf.predict(X_test)
#
# # calculate metrics
# print("\n")
#
# print("Classification Report: ")
# print(classification_report(y_test,y_pred_svm))
# print("\n")
#
# print("Accuracy : ", accuracy_score(y_test, y_pred_svm) * 100)
# print("\n")
#
# #KNN
# # standardize the data
# stdsc = StandardScaler()
#
# stdsc.fit(X_train)
#
# X_train_std = stdsc.transform(X_train)
# X_test_std = stdsc.transform(X_test)
#
# # perform training
# # creating the classifier object
# clf_knn = KNeighborsClassifier(n_neighbors=3)
#
# # performing training
# clf_knn.fit(X_train_std, y_train)
#
# #%%-----------------------------------------------------------------------
# # make predictions
#
# # predicton on test
# y_pred_knn = clf.predict(X_test_std)
#
# #%%-----------------------------------------------------------------------
# # calculate metrics
#
# print("\n")
# print("Classification Report: ")
# print(classification_report(y_test,y_pred_knn))
# print("\n")
#
#
# print("Accuracy : ", accuracy_score(y_test, y_pred_knn) * 100)
# print("\n")
#
# #Naive Bayese
# # creating the classifier object
# clf_nb = GaussianNB()
#
# # performing training
# clf_nb.fit(X_train, y_train)
#
# #%%-----------------------------------------------------------------------
# # make predictions
#
# # predicton on test
# y_pred_nb = clf_nb.predict(X_test)
#
# y_pred_nb_score = clf_nb.predict_proba(X_test)
#
# #%%-----------------------------------------------------------------------
# # calculate metrics
#
# print("\n")
#
# print("Classification Report: ")
# print(classification_report(y_test,y_pred_nb))
# print("\n")
#
#
# print("Accuracy : ", accuracy_score(y_test, y_pred_nb) * 100)
# print("\n")
#
# print("ROC_AUC : ", roc_auc_score(y_test,y_pred_nb_score[:,1]) * 100)
# print("\n")
#







# =================================================================
# GUI
# =================================================================
