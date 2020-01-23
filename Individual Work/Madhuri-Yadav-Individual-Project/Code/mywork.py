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

#set seed
seed = 100


#Extracting Month from release date
merged_inner['release_date_temp'] = pd.to_datetime(merged_inner['release_date'],format='%Y-%m-%d', errors='coerce')  #Converting string to datetime
merged_inner['release_month'] = pd.to_datetime(merged_inner['release_date_temp']).dt.month #extracting month from datetime(Releasedate) column
#df_cleaned['release_month'] = pd.to_numeric(df_cleaned['release_month'],errors='coerce') #converting float to int
merged_inner['release_month'] = merged_inner['release_month'].astype('category') #changing type from num to category
merged_inner = merged_inner.drop(['release_date_temp'], axis=1) #Deleting temp column

# Removing Duplicates
merged_inner.drop_duplicates(inplace = True)     # no duplicates found

# =================================================================
# Modeling
# =================================================================

# Spliting and encoding data
# split the dataset into input and target variables
print("original data : ", len(merged_inner))
X = merged_inner.loc[:,['runtime','averageRating','budget','Genre','Production_Company','release_month', 'popularity']]  #
y = merged_inner.loc[:,['New_status']]


#Scaling numerical values
scaler = MinMaxScaler()
X.loc[:,['runtime','averageRating','budget', 'popularity']]= scaler.fit_transform(X.loc[:,['runtime','averageRating','budget', 'popularity']])

# encloding the class with sklearn's LabelEncoder - one-hot encoding
le = LabelEncoder()

# fit and transform the class
y = le.fit_transform(y)
X = pd.get_dummies(X)


# split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

# Over sampling
# RandomOverSampler (with random_state=0)
ros = RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_sample(X_train, y_train)


# Decision Tree Gini
# perform training with giniIndex.
# creating the classifier object
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=seed, min_samples_leaf=5)

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
clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=seed, min_samples_leaf=5)

# performing training
clf_entropy.fit(X_train, y_train)

# predicton on test using gini
y_pred_entropy = clf_entropy.predict(X_test)

print("Classification Report for DT Entropy: ")
print(classification_report(y_test,y_pred_entropy.ravel()))
print("Accuracy : ", accuracy_score(y_test, y_pred_entropy) * 100)


#Random Forest
# specify random forest classifier
clf_rf = RandomForestClassifier(n_estimators=100,random_state=seed)

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

# predicton on test
y_pred_knn = clf.predict(X_test_std)

# calculate metrics

print("Classification Report for KNN: ")
print(classification_report(y_test,y_pred_knn))
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


# calculate metrics
print("Classification Report for NB: ")
print(classification_report(y_test,y_pred_nb))
print("Accuracy : ", accuracy_score(y_test, y_pred_nb) * 100)
print("\n")

#Ensembling
final_pred = np.array([])
for i in range(0,len(X_test)):
    final_pred = np.append(final_pred, mode([y_pred_rf[i], y_pred_svm[i], y_pred_entropy[i]]))
    
#Applying ADA Boosting
classifier = AdaBoostClassifier(RandomForestClassifier(n_estimators=100,random_state=seed),n_estimators=100,random_state=seed)
classifier.fit(X_train, y_train)

# predicton on test using all features
y_pred_boost = classifier.predict(X_test)

print("Classification Report for boosting: ")
print(classification_report(y_test,y_pred_boost))
print("Accuracy : ", accuracy_score(y_test, y_pred_boost) * 100)


print("*"*50)
print("Accuracy DT Gini : ", accuracy_score(y_test, y_pred_gini) * 100)
print("Accuracy DT Entropy: ", accuracy_score(y_test, y_pred_entropy) * 100)
print("Accuracy SVM: ", accuracy_score(y_test, y_pred_svm) * 100)
print("Accuracy RF: ", accuracy_score(y_test, y_pred_rf) * 100)
print("Accuracy KNN: ", accuracy_score(y_test, y_pred_knn) * 100)
print("Accuracy NB: ", accuracy_score(y_test, y_pred_nb) * 100)
print("Accuracy Bagging with Mode method: ", accuracy_score(y_test, final_pred) * 100)
print("Accuracy ADA: ", accuracy_score(y_test, y_pred_boost) * 100)
print("*"*50)

#Printing results for our best model
print("ROC_AUC : ", roc_auc_score(y_test, y_pred_boost) * 100)
print("Accuracy K: ", cohen_kappa_score(y_test, y_pred_boost)* 100)

# ROC Graph
y_pred_score = classifier.predict_proba(X_test)
preds = y_pred_score[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# confusion matrix for AdaBoosting
conf_matrix = confusion_matrix(y_test, y_pred_rf)
class_names = merged_inner['New_status'].unique()

df_cm = pd.DataFrame(conf_matrix, index=class_names, columns=class_names )

plt.figure(figsize=(5,5))

hm = sns.heatmap(df_cm, cbar=False, annot=True, square=True, fmt='d', annot_kws={'size': 20}, yticklabels=df_cm.columns, xticklabels=df_cm.columns)

hm.yaxis.set_ticklabels(hm.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
hm.xaxis.set_ticklabels(hm.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=20)
plt.ylabel('True label',fontsize=20)
plt.xlabel('Predicted label',fontsize=20)
plt.title("Confusion Matrix AdaBoost Model")
plt.tight_layout()
plt.show()
