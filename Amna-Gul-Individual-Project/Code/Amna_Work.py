import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro


from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split # Import train_test_split function
from imblearn.over_sampling import RandomOverSampler #For over sampling
from sklearn import metrics
import sklearn.metrics as metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics.classification import cohen_kappa_score
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
import os
os.environ["PATH"] += os.pathsep + '/Users/amnagul/anaconda3/pkgs/graphviz-2.40.1-hefbbd9a_2/bin/'
# #%%-----------------------------------------------------------------------

# Libraries for GUI
import sys
from PyQt5.QtWidgets import (QMainWindow, QApplication, QWidget, QPushButton, QAction, QComboBox, QLabel,
                             QGridLayout, QCheckBox, QGroupBox, QVBoxLayout, QHBoxLayout, QLineEdit, QPlainTextEdit)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import Qt
from scipy import interp
from itertools import cycle
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QSizePolicy, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

#set seed
seed = 100

# reading csv file data
movie_data_orig = pd.read_csv('movies_metadata.csv')


# removing 10 irrelevant columns
df_cleaned = movie_data_orig.drop(["adult", "belongs_to_collection", "homepage", "original_title", "overview", "poster_path", "production_countries",
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
cols = ['budget', 'id', 'imdb_id', 'popularity', 'original_language', 'production_companies', 'release_date', 'revenue', 'runtime', 'title', 'vote_average', 'vote_count', 'status', 'New_status', 'genres']
df_cleaned = df_cleaned[cols]


# rearranging columns of dataframe
cols = df_cleaned.columns.tolist()
# Setting Production_companies as last col for easier manipulation
cols = ['budget', 'imdb_id', 'popularity', 'original_language', 'release_date', 'revenue', 'runtime', 'title', 'vote_average', 'vote_count', 'status', 'New_status', 'Genre', 'production_companies']
df_cleaned = df_cleaned[cols]


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
cols = ['budget', 'imdb_id', 'popularity', 'release_date', 'startYear', 'release_month', 'revenue', 'runtime', 'title', 'Genre', 'Production_Company', 'Director', 'averageRating', 'numVotes', 'original_language','status', 'New_status',]
merged_inner = merged_inner[cols]
merged_inner["startYear"].min()

len(merged_inner.original_language.unique())     # 1173


# finding missing values
# a = merged_inner.isnull().sum()           # returns 0 for each column meaning no missing values

merged_inner.dtypes       # release_date is of object (i.e. string data type) instead of datetime
merged_inner['release_date'] =  pd.to_datetime(merged_inner['release_date'])    # converting release_date to datetime object
merged_inner['startYear'] = merged_inner['startYear'].astype(str).astype(int)     # converting startYear to int instead of object

len(merged_inner)    # 2222

# Removing Duplicates
merged_inner.drop_duplicates(inplace = True)     # no duplicates btw
#
# merged_inner.to_csv(r"Cleaned_df.csv", index=None, header=True)

# =================================================================
# GUI
# =================================================================
#

merged_inner.to_csv("GUI_df.csv")
gui_df = pd.read_csv("GUI_df.csv")

font_size_window = 'font-size:15px'



class DecisionTree(QMainWindow):
    #::----------------------
    # Implementation of Decision Tree Algorithm using the movie dataset
    # the methods in this class are
    #       _init_ : initialize the class
    #       initUi : creates the canvas and all the elements in the canvas
    #       update : populates the elements of the canvas base on the parameters
    #               chosen by the user
    #       view_tree : shows the tree in a pdf form
    #::----------------------

    send_fig = pyqtSignal(str)

    def __init__(self):
        super(DecisionTree, self).__init__()

        self.Title ="Decision Tree Classifier"
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the element to create a dashboard with
        #  all the necessary elements to present the results from the algorithm
        #  The canvas is divided using a  grid loyout to facilitate the drawing
        #  of the elements
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QGridLayout(self.main_widget)

        self.groupBox1 = QGroupBox('ML Decision Tree Features')
        self.groupBox1Layout= QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(features_list[0],self)
        self.feature1 = QCheckBox(features_list[1],self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4],self)
        self.feature5 = QCheckBox(features_list[5],self)
        self.feature6 = QCheckBox(features_list[6], self)
        # self.feature7 = QCheckBox(features_list[7], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)
        self.feature6.setChecked(True)
        # self.feature7.setChecked(True)

        self.lblPercentTest = QLabel('Percentage for Test :')
        self.lblPercentTest.adjustSize()

        self.txtPercentTest = QLineEdit(self)
        self.txtPercentTest.setText("30")

        self.lblMaxDepth = QLabel('Maximun Depth :')
        self.txtMaxDepth = QLineEdit(self)
        self.txtMaxDepth.setText("3")

        self.btnExecute = QPushButton("Execute DT")
        self.btnExecute.clicked.connect(self.update)

        self.btnDTFigure = QPushButton("View Tree")
        self.btnDTFigure.clicked.connect(self.view_tree)

        # We create a checkbox for each feature

        self.groupBox1Layout.addWidget(self.feature0,0,0)
        self.groupBox1Layout.addWidget(self.feature1,0,1)
        self.groupBox1Layout.addWidget(self.feature2,1,0)
        self.groupBox1Layout.addWidget(self.feature3,1,1)
        self.groupBox1Layout.addWidget(self.feature4,2,0)
        self.groupBox1Layout.addWidget(self.feature5,2,1)
        self.groupBox1Layout.addWidget(self.feature6,3,0)
        # self.groupBox1Layout.addWidget(self.feature7,3,1)
        self.groupBox1Layout.addWidget(self.lblPercentTest,4,0)
        self.groupBox1Layout.addWidget(self.txtPercentTest,4,1)
        self.groupBox1Layout.addWidget(self.lblMaxDepth,5,0)
        self.groupBox1Layout.addWidget(self.txtMaxDepth,5,1)
        self.groupBox1Layout.addWidget(self.btnExecute,6,0)
        self.groupBox1Layout.addWidget(self.btnDTFigure,6,1)

        self.groupBox2 = QGroupBox('Results from the model')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.lblResults = QLabel('Results:')
        self.lblResults.adjustSize()
        self.txtResults = QPlainTextEdit()
        self.lblAccuracy = QLabel('Accuracy:')
        self.txtAccuracy = QLineEdit()

        self.groupBox2Layout.addWidget(self.lblResults)
        self.groupBox2Layout.addWidget(self.txtResults)
        self.groupBox2Layout.addWidget(self.lblAccuracy)
        self.groupBox2Layout.addWidget(self.txtAccuracy)

        #::-------------------------------------
        # Graphic 1 : Confusion Matrix
        #::-------------------------------------

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes=[self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBoxG1 = QGroupBox('Confusion Matrix')
        self.groupBoxG1Layout= QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)

        self.groupBoxG1Layout.addWidget(self.canvas)

        #::--------------------------------------------
        ## End Graph1
        #::--------------------------------------------

        #::---------------------------------------------
        # Graphic 2 : ROC Curve
        #::---------------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('ROC Curve')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)

        self.groupBoxG2Layout.addWidget(self.canvas2)

        #::---------------------------------------------------
        # Graphic 3 : ROC Curve by Class
        #::---------------------------------------------------

        self.fig3 = Figure()
        self.ax3 = self.fig3.add_subplot(111)
        self.axes3 = [self.ax3]
        self.canvas3 = FigureCanvas(self.fig3)

        self.canvas3.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas3.updateGeometry()

        self.groupBoxG3 = QGroupBox('ROC Curve by Class')
        self.groupBoxG3Layout = QVBoxLayout()
        self.groupBoxG3.setLayout(self.groupBoxG3Layout)

        self.groupBoxG3Layout.addWidget(self.canvas3)

        ## End of elements o the dashboard

        self.layout.addWidget(self.groupBox1,0,0)
        self.layout.addWidget(self.groupBoxG1,0,1)
        self.layout.addWidget(self.groupBox2,0,2)
        self.layout.addWidget(self.groupBoxG2,1,1)
        self.layout.addWidget(self.groupBoxG3,1,2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()


    def update(self):
        '''
        Decision Tree Algorithm
        We pupulate the dashboard using the parametres chosen by the user
        The parameters are processed to execute in the skit-learn Decision Tree algorithm
          then the results are presented in graphics and reports in the canvas
        :return: None
        '''

        # We process the parameters
        self.list_dt = pd.DataFrame([])
        if self.feature0.isChecked():
            if len(self.list_dt)==0:
                self.list_dt = gui_df[dt_features[0]]
            else:
                self.list_dt = pd.concat([self.list_dt, gui_df[dt_features[0]]],axis=1)

        if self.feature1.isChecked():
            if len(self.list_dt) == 0:
                self.list_dt = gui_df[features_list[1]]
            else:
                self.list_dt = pd.concat([self.list_dt, gui_df[dt_features[1]]],axis=1)

        if self.feature2.isChecked():
            if len(self.list_dt) == 0:
                self.list_dt = gui_df[dt_features[2]]
            else:
                self.list_dt = pd.concat([self.list_dt, gui_df[dt_features[2]]],axis=1)

        if self.feature3.isChecked():
            if len(self.list_dt) == 0:
                self.list_dt = gui_df[dt_features[3]]
            else:
                self.list_dt = pd.concat([self.list_dt, gui_df[dt_features[3]]],axis=1)

        if self.feature4.isChecked():
            if len(self.list_dt) == 0:
                self.list_dt = gui_df[dt_features[4]]
            else:
                self.list_dt = pd.concat([self.list_dt, gui_df[dt_features[4]]],axis=1)

        if self.feature5.isChecked():
            if len(self.list_dt) == 0:
                self.list_dt = gui_df[dt_features[5]]
            else:
                self.list_dt = pd.concat([self.list_dt, gui_df[dt_features[5]]],axis=1)

        if self.feature6.isChecked():
            if len(self.list_dt) == 0:
                self.list_dt = gui_df[dt_features[6]]
            else:
                self.list_dt = pd.concat([self.list_dt, gui_df[dt_features[6]]],axis=1)

        # if self.feature7.isChecked():
        #     if len(self.list_corr_features) == 0:
        #         self.list_corr_features = ff_happiness[features_list[7]]
        #     else:
        #         self.list_corr_features = pd.concat([self.list_corr_features, ff_happiness[features_list[7]]],axis=1)


        vtest_per = float(self.txtPercentTest.text())
        vmax_depth = float(self.txtMaxDepth.text())

        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        self.txtResults.clear()
        self.txtResults.setUndoRedoEnabled(False)

        vtest_per = vtest_per / 100





        X_dt =  self.list_dt
        y_dt = gui_df['New_status']

        scaler = MinMaxScaler()
        X_dt.loc[:,['runtime','averageRating','budget', 'popularity']]= scaler.fit_transform(X_dt.loc[:,['runtime','averageRating','budget', 'popularity']])

        class_le = LabelEncoder()

        # fit and transform the class


        y_dt = class_le.fit_transform(y_dt)
        X_dt = pd.get_dummies(X_dt)



        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=vtest_per, random_state=100, stratify=y_dt)

        # perform training with entropy.
        # Decision tree with entropy
        self.clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=vmax_depth, min_samples_leaf=5)


        # Performing training
        self.clf_entropy.fit(X_train, y_train)


        # predicton on test using entropy
        y_pred_entropy = self.clf_entropy.predict(X_test)


        # confusion matrix for entropy model

        conf_matrix = confusion_matrix(y_test, y_pred_entropy.ravel())

        # clasification report

        self.ff_class_rep = classification_report(y_test, y_pred_entropy.ravel())
        self.txtResults.appendPlainText(self.ff_class_rep)

        # accuracy score

        self.ff_accuracy_score = accuracy_score(y_test, y_pred_entropy) * 100
        self.txtAccuracy.setText(str(self.ff_accuracy_score))


        #::----------------------------------------------------------------
        # Graph1 -- Confusion Matrix
        #::-----------------------------------------------------------------

        self.ax1.set_xlabel('Predicted label')
        self.ax1.set_ylabel('True label')

        class_names1 = ['','0', '1']
        # class_names1 = merged_inner['New_status'].unique()

        self.ax1.matshow(conf_matrix, cmap= plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(class_names1)
        self.ax1.set_xticklabels(class_names1,rotation = 90)

        for i in range(len(class_names)):
            for j in range(len(class_names)):
                y_pred_score = self.clf_entropy.predict_proba(X_test)
                self.ax1.text(j, i, str(conf_matrix[i][j]))

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

        #::-----------------------------------------------------
        # End Graph 1 -- Confusioin Matrix
        #::-----------------------------------------------------

        #::-----------------------------------------------------
        # Graph 2 -- ROC Cure
        #::-----------------------------------------------------

        y_test_bin = label_binarize(y_test, classes=[0, 1])
        n_classes = y_test_bin.shape[1]

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_score.ravel())

        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        lw = 2
        self.ax2.plot(fpr[2], tpr[2], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        self.ax2.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        self.ax2.set_xlim([0.0, 1.0])
        self.ax2.set_ylim([0.0, 1.05])
        self.ax2.set_xlabel('False Positive Rate')
        self.ax2.set_ylabel('True Positive Rate')
        self.ax2.set_title('ROC Curve Decision Tree')
        self.ax2.legend(loc="lower right")

        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

        #::--------------------------------
        ### Graph 3 Roc Curve by class
        #::--------------------------------

        str_classes= ['Success','Failure']
        colors = cycle(['magenta', 'darkorange'])
        for i, color in zip(range(n_classes), colors):
            self.ax3.plot(fpr[i], tpr[i], color=color, lw=lw,
                     label='{0} (area = {1:0.2f})'
                           ''.format(str_classes[i], roc_auc[i]))

        self.ax3.plot([0, 1], [0, 1], 'k--', lw=lw)
        self.ax3.set_xlim([0.0, 1.0])
        self.ax3.set_ylim([0.0, 1.05])
        self.ax3.set_xlabel('False Positive Rate')
        self.ax3.set_ylabel('True Positive Rate')
        self.ax3.set_title('ROC Curve by Class')
        self.ax3.legend(loc="lower right")

        # show the plot
        self.fig3.tight_layout()
        self.fig3.canvas.draw_idle()


    def view_tree(self):
        '''
        Executes the graphviz to create a tree view of the information
         then it presents the graphic in a pdf formt using webbrowser
        :return:None
        '''
        dot_data = export_graphviz(self.clf_entropy, filled=True, rounded=True, class_names=class_names,
                                   feature_names=self.list_dt.columns, out_file=None)


        graph = graph_from_dot_data(dot_data)
        graph.write_pdf("decision_tree_entropy.pdf")
        webbrowser.open_new(r'decision_tree_entropy.pdf')



class CorrelationPlot(QMainWindow):
    # ;:-----------------------------------------------------------------------
    # This class creates a canvas to draw a correlation plot
    # It presents all the features plus the happiness score
    # the methods for this class are:
    #   _init_
    #   initUi
    #   update
    #::-----------------------------------------------------------------------
    send_fig = pyqtSignal(str)

    def __init__(self):
        #::--------------------------------------------------------
        # Initialize the values of the class
        #::--------------------------------------------------------
        super(CorrelationPlot, self).__init__()

        self.Title = 'Correlation Plot'
        self.initUi()

    def initUi(self):
        #::--------------------------------------------------------------
        #  Creates the canvas and elements of the canvas
        #::--------------------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.main_widget = QWidget(self)

        self.layout = QVBoxLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Correlation Plot Features')
        self.groupBox1Layout = QGridLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.feature0 = QCheckBox(features_list[0], self)
        self.feature1 = QCheckBox(features_list[1], self)
        self.feature2 = QCheckBox(features_list[2], self)
        self.feature3 = QCheckBox(features_list[3], self)
        self.feature4 = QCheckBox(features_list[4], self)
        self.feature5 = QCheckBox(features_list[5], self)
        self.feature6 = QCheckBox(features_list[6], self)
        # self.feature7 = QCheckBox(features_list[7], self)
        self.feature0.setChecked(True)
        self.feature1.setChecked(True)
        self.feature2.setChecked(True)
        self.feature3.setChecked(True)
        self.feature4.setChecked(True)
        self.feature5.setChecked(True)
        self.feature6.setChecked(True)
        # self.feature7.setChecked(True)

        self.btnExecute = QPushButton("Create Plot")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.feature0, 0, 0)
        self.groupBox1Layout.addWidget(self.feature1, 0, 1)
        self.groupBox1Layout.addWidget(self.feature2, 0, 2)
        self.groupBox1Layout.addWidget(self.feature3, 0, 3)
        self.groupBox1Layout.addWidget(self.feature4, 1, 0)
        self.groupBox1Layout.addWidget(self.feature5, 1, 1)
        self.groupBox1Layout.addWidget(self.feature6, 1, 2)
        # self.groupBox1Layout.addWidget(self.feature7, 1, 3)
        self.groupBox1Layout.addWidget(self.btnExecute, 2, 0)

        self.fig = Figure()
        self.ax1 = self.fig.add_subplot(111)
        self.axes = [self.ax1]
        self.canvas = FigureCanvas(self.fig)

        self.canvas.setSizePolicy(QSizePolicy.Expanding,
                                  QSizePolicy.Expanding)

        self.canvas.updateGeometry()

        self.groupBox2 = QGroupBox('Correlation Plot')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.groupBox2Layout.addWidget(self.canvas)

        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBox2)

        self.setCentralWidget(self.main_widget)
        self.resize(900, 700)
        self.show()
        self.update()

    def update(self):

        #::------------------------------------------------------------
        # Populates the elements in the canvas using the values
        # chosen as parameters for the correlation plot
        #::------------------------------------------------------------
        self.ax1.clear()

        # X_1 = gui_df["Happiness.Score"]

        list_corr_features = pd.DataFrame()
        if self.feature0.isChecked():
            list_corr_features = pd.concat([list_corr_features, gui_df[features_list[0]]], axis=1)

        if self.feature1.isChecked():
            list_corr_features = pd.concat([list_corr_features, gui_df[features_list[1]]], axis=1)

        if self.feature2.isChecked():
            list_corr_features = pd.concat([list_corr_features, gui_df[features_list[2]]], axis=1)

        if self.feature3.isChecked():
            list_corr_features = pd.concat([list_corr_features, gui_df[features_list[3]]], axis=1)
        if self.feature4.isChecked():
            list_corr_features = pd.concat([list_corr_features, gui_df[features_list[4]]], axis=1)

        if self.feature5.isChecked():
            list_corr_features = pd.concat([list_corr_features, gui_df[features_list[5]]], axis=1)

        if self.feature6.isChecked():
            list_corr_features = pd.concat([list_corr_features, gui_df[features_list[6]]], axis=1)

        # if self.feature7.isChecked():
        #     list_corr_features = pd.concat([list_corr_features, gui_df[features_list[7]]], axis=1)

        vsticks = ["dummy"]
        vsticks1 = list(list_corr_features.columns)
        vsticks1 = vsticks + vsticks1
        res_corr = list_corr_features.corr()
        self.ax1.matshow(res_corr, cmap=plt.cm.get_cmap('Blues', 14))
        self.ax1.set_yticklabels(vsticks1)
        self.ax1.set_xticklabels(vsticks1, rotation=90)

        self.fig.tight_layout()
        self.fig.canvas.draw_idle()


class PlotCanvas(FigureCanvas):
    #::----------------------------------------------------------
    # creates a figure on the canvas
    # later on this element will be used to draw a histogram graph
    #::----------------------------------------------------------
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self):
        self.ax = self.figure.add_subplot(111)


class CanvasWindow(QMainWindow):
    #::----------------------------------
    # Creates a canvas containing the plot for the initial analysis
    # ;;----------------------------------
    def __init__(self, parent=None):
        super(CanvasWindow, self).__init__(parent)

        self.left = 200
        self.top = 200
        self.Title = 'Distribution'
        self.width = 1000
        self.height = 1000
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.setGeometry(self.left, self.top, self.width, self.height)

        self.m = PlotCanvas(self, width=8, height=6)
        self.m.move(0, 30)


class App(QMainWindow):
    #::-------------------------------------------------------
    # This class creates all the elements of the application
    #::-------------------------------------------------------

    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.Title = 'Predicting Movie Success/Failure via ML'
        self.width = 500
        self.height = 300
        self.initUI()

    def initUI(self):
        #::-------------------------------------------------
        # Creates the manu and the items
        #::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        #::-----------------------------
        # Create the menu bar
        # and three items for the menu, File, EDA Analysis and ML Models
        #::-----------------------------
        mainMenu = self.menuBar()
        mainMenu.setStyleSheet('background-color: lightblue')

        fileMenu = mainMenu.addMenu('File')
        EDAMenu = mainMenu.addMenu('EDA Analysis')
        MLModelMenu = mainMenu.addMenu('ML Models')

        #::--------------------------------------
        # Exit application
        # Creates the actions for the fileMenu item
        #::--------------------------------------

        exitButton = QAction(QIcon('enter.png'), 'Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        #::----------------------------------------
        # EDA analysis
        # Creates the actions for the EDA Analysis item
        # Initial Assesment : Histogram about the level of happiness in 2017
        # Happiness Final : Presents the correlation between the index of happiness and a feature from the datasets.
        # Correlation Plot : Correlation plot using all the dims in the datasets
        #::----------------------------------------

        EDA1Button = QAction(QIcon('analysis.png'), 'Initial Assesment', self)
        EDA1Button.setStatusTip('Presents the initial datasets')
        EDA1Button.triggered.connect(self.EDA1)
        EDAMenu.addAction(EDA1Button)

        # EDA2Button = QAction(QIcon('analysis.png'), 'Genre Count', self)
        # EDA2Button.setStatusTip('Final Happiness Graph')
        # EDA2Button.triggered.connect(self.EDA2)
        # EDAMenu.addAction(EDA2Button)

        EDA4Button = QAction(QIcon('analysis.png'), 'Correlation Plot', self)
        EDA4Button.setStatusTip('Features Correlation Plot')
        EDA4Button.triggered.connect(self.EDA4)
        EDAMenu.addAction(EDA4Button)

        #::--------------------------------------------------
        # ML Models for prediction
        # There are two models
        #       Decision Tree
        #       Random Forest
        #::--------------------------------------------------
        # Decision Tree Model
        #::--------------------------------------------------
        MLModel1Button = QAction(QIcon(), 'Decision Tree Entropy', self)
        MLModel1Button.setStatusTip('ML algorithm with Entropy ')
        MLModel1Button.triggered.connect(self.MLDT)

        #::------------------------------------------------------
        # Random Forest Classifier
        #::------------------------------------------------------
        MLModel2Button = QAction(QIcon(), 'Random Forest Classifier', self)
        MLModel2Button.setStatusTip('Random Forest Classifier ')
        MLModel2Button.triggered.connect(self.MLRF)

        MLModelMenu.addAction(MLModel1Button)
        MLModelMenu.addAction(MLModel2Button)

        self.dialogs = list()

    def EDA1(self):
        #::------------------------------------------------------
        # Creates the graph for number of movies per Genre

        #::------------------------------------------------------
        dialog = CanvasWindow(self)
        dialog.m.plot()
        x = gui_df["New_status"].unique()
        y = gui_df['New_status'].value_counts()
        dialog.m.ax.bar(y.index,  y.values )
        dialog.m.ax.set_title('Success/Flop Movie Count')
        dialog.m.ax.set_xlabel("Classes")
        dialog.m.ax.set_ylabel("Count")
        dialog.m.ax.grid(True)
        dialog.m.draw()
        self.dialogs.append(dialog)
        dialog.show()

    # def EDA2(self):
    #     #::---------------------------------------------------------
    #     # This function creates an instance of HappinessGraphs class
    #     # This class creates a graph using the features in the dataset
    #     # happiness vrs the score of happiness
    #     #::---------------------------------------------------------
    #     dialog = HappinessGraphs()
    #     self.dialogs.append(dialog)
    #     dialog.show()

    def EDA4(self):
        #::----------------------------------------------------------
        # This function creates an instance of the CorrelationPlot class
        #::----------------------------------------------------------
        dialog = CorrelationPlot()
        self.dialogs.append(dialog)
        dialog.show()

    def MLDT(self):
        #::-----------------------------------------------------------
        # This function creates an instance of the DecisionTree class
        # This class presents a dashboard for a Decision Tree Algorithm
        # using the happiness dataset
        #::-----------------------------------------------------------
        dialog = DecisionTree()
        self.dialogs.append(dialog)
        dialog.show()

    def MLRF(self):
        #::-------------------------------------------------------------
        # This function creates an instance of the Random Forest Classifier Algorithm
        # using the happiness dataset
        #::-------------------------------------------------------------
        dialog = RandomForest()
        self.dialogs.append(dialog)
        dialog.show()


def main():
    #::-------------------------------------------------
    # Initiates the application
    #::-------------------------------------------------
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    ex = App()
    ex.show()
    sys.exit(app.exec_())


def movie_prediction():
    #::--------------------------------------------------
    # Loads the dataset movies_metadata.csv ( Raw/Original dataset)
    # Loads the dataset cleaned_df.csv
    #::--------------------------------------------------

    global gui_df
    global features_list
    global class_names
    global dt_features

    gui_df = pd.read_csv('GUI_df.csv')
    features_list = ['budget', 'startYear', 'revenue', 'runtime', 'popularity', 'averageRating', 'numVotes','status']
    class_names = ['0', '1']

    dt_features = ['runtime', 'averageRating', 'budget', 'Genre', 'Production_Company', 'release_month', 'popularity']


if __name__ == '__main__':
    #::------------------------------------
    # First reads the data then calls for the application
    #::------------------------------------
    movie_prediction()
    main()

