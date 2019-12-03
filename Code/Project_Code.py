import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from statistics import mode
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
# Libraries to display decision tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
import webbrowser

# #%%-----------------------------------------------------------------------
# import os
# os.environ["PATH"] += os.pathsep + '/Graphviz2.38/bin'
# #%%-----------------------------------------------------------------------
#
# # Libraries for GUI
# import sys
# from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
#                              QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit)
# from PyQt5.QtGui import QIcon
# from PyQt5.QtCore import pyqtSlot
# from PyQt5.QtCore import pyqtSignal
# from PyQt5.QtCore import Qt
# from scipy import interp
# from itertools import cycle
# from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy, QMessageBox
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
# from matplotlib.figure import Figure



# reading csv file data
import pandas as pd
movie_data_orig = pd.read_csv(r"C:\Users\Madhuri Yadav\Downloads\Final-Project-Group8-master\Final-Project-Group8-master\Code\movies_metadata.csv")
# print(movie_data_orig)     # [45466 rows x 24 columns]

# removing 12 irrelevant columns
df_cleaned = movie_data_orig.drop(["adult", "belongs_to_collection", "homepage", "original_language",
                                   "original_title", "overview", "poster_path", "production_countries",
                                   "spoken_languages", "status", "tagline", "video" ], axis=1)

# print(df_cleaned.columns)   #  'budget', 'genres', 'id', 'imdb_id' 'popularity', 'production_companies', 'release_date', 'revenue', 'runtime', 'title', 'vote_average', 'vote_count']
#df_cleaned.dtypes       # release_date is of object (i.e. string data type) instead of datetime

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
df_cleaned.rename(columns = {'name' : 'Production_Company'}, inplace = True)
df_cleaned = df_cleaned[~df_cleaned['Production_Company'].isnull()]
len(df_cleaned.Production_Company.unique())

# Adding Director col using imdb files
dir_id_imdb = pd.read_csv(r'C:\Users\Madhuri Yadav\Downloads\Final-Project-Group8-master\Final-Project-Group8-master\Code\title_crew.tsv', sep='\t')
merged_inner = pd.merge(left=df_cleaned,right=dir_id_imdb, left_on='imdb_id', right_on='tconst')
dir_name_imdb = pd.read_csv(r'C:\Users\Madhuri Yadav\Downloads\Final-Project-Group8-master\Final-Project-Group8-master\Code\name_basics.tsv', sep='\t')
merged_inner = pd.merge(left=merged_inner,right=dir_name_imdb, left_on='directors', right_on='nconst')
merged_inner = merged_inner.drop(["tconst", "directors", "nconst"], axis=1)     # removing irrelevant cols
merged_inner.rename(columns = {'primaryName' : 'Director'}, inplace = True)


# Adding Avg_ratings & Total votes cols using imdb files
ratings_imdb = pd.read_csv(r'C:\Users\Madhuri Yadav\Downloads\Final-Project-Group8-master\Final-Project-Group8-master\Code\title_ratings.tsv', sep='\t')
merged_inner = pd.merge(left=merged_inner,right=ratings_imdb, left_on='imdb_id', right_on='tconst')
merged_inner = merged_inner.drop(["tconst", "vote_average", "vote_count"], axis=1)     # removing old vote_avg/count cols

# Adding Movie release year column from imdb file
releaseYr_imdb = pd.read_csv(r'C:\Users\Madhuri Yadav\Downloads\Final-Project-Group8-master\Final-Project-Group8-master\Code\title_year.tsv', sep='\t')
merged_inner = pd.merge(left=merged_inner,right=releaseYr_imdb, left_on='imdb_id', right_on='tconst')
merged_inner = merged_inner.drop(["tconst"], axis=1)
cols = merged_inner.columns.tolist()
# Setting StartYear col beside release_date col
cols = ['budget', 'imdb_id', 'popularity', 'release_date', 'startYear', 'revenue', 'runtime', 'title', 'status', 'New_status', 'Genre', 'Production_Company', 'Director', 'averageRating', 'numVotes']
merged_inner = merged_inner[cols]
merged_inner["startYear"].min()

len(merged_inner.Director.unique())     # 1173

len(merged_inner)
merged_inner.dtypes       # release_date is of object (i.e. string data type) instead of datetime

#Extracting Month from release date
df_cleaned['release_date_temp'] = pd.to_datetime(df_cleaned['release_date'],format='%Y-%m-%d', errors='coerce')  #Converting string to datetime
df_cleaned['release_month'] = pd.to_datetime(df_cleaned['release_date_temp']).dt.month #extracting month from datetime(Releasedate) column
#df_cleaned['release_month'] = pd.to_numeric(df_cleaned['release_month'],errors='coerce') #converting float to int
df_cleaned['release_month'] = df_cleaned['release_month'].astype('category')
print(df_cleaned.dtypes)

df_cleaned = df_cleaned.drop(['release_date_temp'], axis=1)

# Removing Duplicates
merged_inner.drop_duplicates(inplace = True)

merged_inner.to_csv(r"Cleaned_df.csv", index=None, header=True)

# =================================================================
# EDA
# =================================================================



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
#
# # Pair Plot
# df_x = df_cleaned[['budget','revenue','runtime','vote_average','vote_count','New_status']]
# sns.set(style = 'ticks')
# sns.pairplot(df_x, hue = 'New_status')
# plt.show()



# =================================================================
# Modeling
# =================================================================

# Decision Tree Gini
#split the dataset into input and target variables

X = df_cleaned.loc[:,['runtime','vote_average','Genre','Production_Company','release_month']]  #
y = df_cleaned.loc[:,['New_status']]

scaler = MinMaxScaler()
X.loc[:,['runtime','vote_average']]= scaler.fit_transform(X.loc[:,['runtime','vote_average']])

# encloding the class with sklearn's LabelEncoder
le = LabelEncoder()

# fit and transform the class
y = le.fit_transform(y)
X = pd.get_dummies(X)

# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

# perform training with giniIndex.
# creating the classifier object
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, min_samples_leaf=5)

# performing training
clf_gini.fit(X_train, y_train)

# predicton on test using gini
y_pred_gini = clf_gini.predict(X_test)

print("Classification Report For DT Gini: ")
print(classification_report(y_test,y_pred_gini))
print("Accuracy : ", accuracy_score(y_test, y_pred_gini.ravel()) * 100)

#Decision Tree Entropy
# perform training with Entropy.
# creating the classifier object
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, min_samples_leaf=5)

# performing training
clf_entropy.fit(X_train, y_train)

# predicton on test using gini
y_pred_entropy = clf_entropy.predict(X_test)

print("Classification Report for DT Entropy: ")
print(classification_report(y_test,y_pred_entropy.ravel()))
print("Accuracy : ", accuracy_score(y_test, y_pred_entropy) * 100)


#Random Forest
# specify random forest classifier
clf_rf = RandomForestClassifier(n_estimators=100)

# perform training
clf_rf.fit(X_train, y_train)

# predicton on test using all features
y_pred_rf = clf_rf.predict(X_test)
y_pred_score = clf_rf.predict_proba(X_test)

print("Classification Report for DT Entropy: ")
print(classification_report(y_test,y_pred_rf))
print("Accuracy : ", accuracy_score(y_test, y_pred_rf) * 100)

#Applying SVM Classification
# perform training
# creating the classifier object
clf = SVC(kernel="linear")

# performing training
clf.fit(X_train, y_train)

# predicton on test
y_pred_svm = clf.predict(X_test)

# calculate metrics
print("\n")

print("Classification Report for SVM:")
print(classification_report(y_test,y_pred_svm))
print("\n")

print("Accuracy : ", accuracy_score(y_test, y_pred_svm) * 100)
print("\n")

#KNN
# standardize the data
stdsc = StandardScaler()

stdsc.fit(X_train)

X_train_std = stdsc.transform(X_train)
X_test_std = stdsc.transform(X_test)

# perform training
# creating the classifier object
clf_knn = KNeighborsClassifier(n_neighbors=3)

# performing training
clf_knn.fit(X_train_std, y_train)

#%%-----------------------------------------------------------------------
# make predictions

# predicton on test
y_pred_knn = clf.predict(X_test_std)

#%%-----------------------------------------------------------------------
# calculate metrics

print("\n")
print("Classification Report for KNN: ")
print(classification_report(y_test,y_pred_knn))
print("\n")


print("Accuracy : ", accuracy_score(y_test, y_pred_knn) * 100)
print("\n")

#Naive Bayese
# creating the classifier object
clf_nb = GaussianNB()

# performing training
clf_nb.fit(X_train, y_train)

#%%-----------------------------------------------------------------------
# make predictions

# predicton on test
y_pred_nb = clf_nb.predict(X_test)

y_pred_nb_score = clf_nb.predict_proba(X_test)

#%%-----------------------------------------------------------------------
# calculate metrics

print("\n")

print("Classification Report for NB: ")
print(classification_report(y_test,y_pred_nb))
print("\n")


print("Accuracy : ", accuracy_score(y_test, y_pred_nb) * 100)
print("\n")

# print("ROC_AUC : ", roc_auc_score(y_test,y_pred_nb_score[:,1]) * 100)
# print("\n")

#Ensembling
final_pred = np.array([])
for i in range(0,len(X_test)):
    final_pred = np.append(final_pred, mode([y_pred_rf[i], y_pred_svm[i], y_pred_knn[i]]))

print("*"*50)
print("Accuracy DT Gini : ", accuracy_score(y_test, y_pred_gini) * 100)
print("Accuracy DT Entropy: ", accuracy_score(y_test, y_pred_entropy) * 100)
print("Accuracy SVM: ", accuracy_score(y_test, y_pred_svm) * 100)
print("Accuracy RF: ", accuracy_score(y_test, y_pred_rf) * 100)
print("Accuracy KNN: ", accuracy_score(y_test, y_pred_knn) * 100)
print("Accuracy NB: ", accuracy_score(y_test, y_pred_nb) * 100)
print("Accuracy final: ", accuracy_score(y_test, final_pred) * 100)

print(y_pred_score)
print("*"*50)


# =================================================================
# GUI
# =================================================================


# class CanvasWindow(QMainWindow):
#     #::----------------------------------
#     # Creates a canvas containing the plot for the initial analysis
#     #;;----------------------------------
#     def __init__(self, parent=None):
#         super(CanvasWindow, self).__init__(parent)
#
#         self.left = 200
#         self.top = 200
#         self.Title = 'Distribution'
#         self.width = 500
#         self.height = 500
#         self.initUI()
#
#     def initUI(self):
#
#         self.setWindowTitle(self.Title)
#         self.setStyleSheet(font_size_window)
#
#         self.setGeometry(self.left, self.top, self.width, self.height)
#
#         self.m = PlotCanvas(self, width=5, height=4)
#         self.m.move(0, 30)
#
# class App(QMainWindow):
#     #::-------------------------------------------------------
#     # This class creates all the elements of the application
#     #::-------------------------------------------------------
#
#     def __init__(self):
#         super().__init__()
#         self.left = 100
#         self.top = 100
#         self.Title = 'Predicting Movie Success/Failure via ML'
#         self.width = 500
#         self.height = 300
#         self.initUI()
#
#     def initUI(self):
#         #::-------------------------------------------------
#         # Creates the manu and the items
#         #::-------------------------------------------------
#         self.setWindowTitle(self.Title)
#         self.setGeometry(self.left, self.top, self.width, self.height)
#
#         #::-----------------------------
#         # Create the menu bar
#         # and three items for the menu, File, EDA Analysis and ML Models
#         #::-----------------------------
#         mainMenu = self.menuBar()
#         mainMenu.setStyleSheet('background-color: lightblue')
#
#         fileMenu = mainMenu.addMenu('File')
#         EDAMenu = mainMenu.addMenu('EDA Analysis')
#         MLModelMenu = mainMenu.addMenu('ML Models')
#
#         #::--------------------------------------
#         # Exit application
#         # Creates the actions for the fileMenu item
#         #::--------------------------------------
#
#         exitButton = QAction(QIcon('enter.png'), 'Exit', self)
#         exitButton.setShortcut('Ctrl+Q')
#         exitButton.setStatusTip('Exit application')
#         exitButton.triggered.connect(self.close)
#
#         fileMenu.addAction(exitButton)
#
#         #::----------------------------------------
#         # EDA analysis
#         # Creates the actions for the EDA Analysis item
#         # Initial Assesment : Histogram about the level of happiness in 2017
#         # Happiness Final : Presents the correlation between the index of happiness and a feature from the datasets.
#         # Correlation Plot : Correlation plot using all the dims in the datasets
#         #::----------------------------------------
#
#         EDA1Button = QAction(QIcon('analysis.png'),'Initial Assesment', self)
#         EDA1Button.setStatusTip('Presents the initial datasets')
#         EDA1Button.triggered.connect(self.EDA1)
#         EDAMenu.addAction(EDA1Button)
#
#         # EDA2Button = QAction(QIcon('analysis.png'), 'Happiness Final', self)
#         # EDA2Button.setStatusTip('Final Happiness Graph')
#         # EDA2Button.triggered.connect(self.EDA2)
#         # EDAMenu.addAction(EDA2Button)
#
#         EDA4Button = QAction(QIcon('analysis.png'), 'Correlation Plot', self)
#         EDA4Button.setStatusTip('Features Correlation Plot')
#         EDA4Button.triggered.connect(self.EDA4)
#         EDAMenu.addAction(EDA4Button)
#
#         #::--------------------------------------------------
#         # ML Models for prediction
#         # There are two models
#         #       Decision Tree
#         #       Random Forest
#         #::--------------------------------------------------
#         # Decision Tree Model
#         #::--------------------------------------------------
#         MLModel1Button =  QAction(QIcon(), 'Decision Tree Entropy', self)
#         MLModel1Button.setStatusTip('ML algorithm with Entropy ')
#         MLModel1Button.triggered.connect(self.MLDT)
#
#         #::------------------------------------------------------
#         # Random Forest Classifier
#         #::------------------------------------------------------
#         MLModel2Button = QAction(QIcon(), 'Random Forest Classifier', self)
#         MLModel2Button.setStatusTip('Random Forest Classifier ')
#         MLModel2Button.triggered.connect(self.MLRF)
#
#         MLModelMenu.addAction(MLModel1Button)
#         MLModelMenu.addAction(MLModel2Button)
#
#         self.dialogs = list()
#
#     def EDA1(self):
#         #::------------------------------------------------------
#         # Creates the graph for number of movies per Genre
#
#         #::------------------------------------------------------
#         dialog = CanvasWindow(self)
#         dialog.m.plot()
#         dialog.m.ax.hist(df_cleaned['Genre'])
#         dialog.m.ax.set_title('Number of Movies per Genre')
#         dialog.m.ax.set_xlabel("Genres")
#         dialog.m.ax.set_ylabel("Frequency")
#         dialog.m.ax.grid(True)
#         dialog.m.draw()
#         self.dialogs.append(dialog)
#         dialog.show()
#
#
#     # def EDA2(self):
#     #     #::---------------------------------------------------------
#     #     # This function creates an instance of HappinessGraphs class
#     #     # This class creates a graph using the features in the dataset
#     #     # happiness vrs the score of happiness
#     #     #::---------------------------------------------------------
#     #     dialog = HappinessGraphs()
#     #     self.dialogs.append(dialog)
#     #     dialog.show()
#
#     def EDA4(self):
#         #::----------------------------------------------------------
#         # This function creates an instance of the CorrelationPlot class
#         #::----------------------------------------------------------
#         dialog = CorrelationPlot()
#         self.dialogs.append(dialog)
#         dialog.show()
#
#     def MLDT(self):
#         #::-----------------------------------------------------------
#         # This function creates an instance of the DecisionTree class
#         # This class presents a dashboard for a Decision Tree Algorithm
#         # using the happiness dataset
#         #::-----------------------------------------------------------
#         dialog = DecisionTree()
#         self.dialogs.append(dialog)
#         dialog.show()
#
#     def MLRF(self):
#         #::-------------------------------------------------------------
#         # This function creates an instance of the Random Forest Classifier Algorithm
#         # using the happiness dataset
#         #::-------------------------------------------------------------
#         dialog = RandomForest()
#         self.dialogs.append(dialog)
#         dialog.show()
#
# def main():
#     #::-------------------------------------------------
#     # Initiates the application
#     #::-------------------------------------------------
#     app = QApplication(sys.argv)
#     app.setStyle('Fusion')
#     ex = App()
#     ex.show()
#     sys.exit(app.exec_())
#
#
# def movie_prediction():
#     #::--------------------------------------------------
#     # Loads the dataset movies_metadata.csv ( Raw/Original dataset)
#     # Loads the dataset cleaned_df.csv
#     #::--------------------------------------------------
#
#     global final_movie
#     global features_list
#     global class_names
#
#     final_movie = pd.read_csv('Cleaned_df.csv')
#     features_list = df_cleaned.columns.tolist()
#     class_names = ['Success', 'Flop']
#
#
# if __name__ == '__main__':
#     #::------------------------------------
#     # First reads the data then calls for the application
#     #::------------------------------------
#     movie_prediction()
#     main()
#
#
