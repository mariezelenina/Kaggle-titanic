#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn as sklearn
from sklearn import metrics
import re as re
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline

from itertools import chain, combinations # to iterate through features

from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectFwe


# ## LOADING DATA
# 
# Xdata_all  <-- Xdata_test          | NONE (testing data, no correct answers given)
#            <-- Xdata_trainval <-   | Ydata_trainval
#                       |     
#                       |---80/20 split 
#                       |  
#                  Xdata_train | Ydata_train
#                  Xdata_val   | Ydata_val
#                               

# In[2]:


Xdata_trainval = pd.read_csv("data/train.csv",index_col=0)
Xdata_test = pd.read_csv("data/test.csv",index_col=0)
data_test_answers = pd.read_csv("data/gender_submission.csv", index_col=0) # sample submission

Ydata_trainval = Xdata_trainval[["Survived"]]

Xdata_trainval_withanswers = Xdata_trainval.copy(deep=True)
Xdata_trainval.drop("Survived", axis=1, inplace=True)
#Xdata_trainval.head()
#Ydata_trainval.head()

Xdata_all = pd.concat([Xdata_trainval, Xdata_test])
Xdata_all.head()

all_xframes = [Xdata_all, Xdata_trainval, Xdata_test, Xdata_trainval_withanswers]


# # FEATURE ENGINEERING
# 
# ## Dealing with missing values

# In[3]:


print('Columns with null values:\n', Xdata_all.isnull().sum())
print("-"*10)


# ### Missing value 1:  Age. Int value. Replacing it by average value (no decimals)

# In[4]:


# Mean age:
Xdata_all["Age"].mean()


# In[5]:


for frame in all_xframes:
    frame["Age"] = frame["Age"].fillna(30)

print('Columns with null values:\n', Xdata_all.isnull().sum())
print("-"*10)


# ### Missing value 2: Fare. Float value. Replace by mean

# In[6]:


Xdata_all["Fare"].mean()


# In[7]:


for frame in all_xframes:
    frame["Fare"] = frame["Fare"].fillna(33.2955)

print('Columns with null values:\n', Xdata_all.isnull().sum())
print("-"*10)


# ### Missing value 3: Cabin. A lot of missing variables. 

# From cabin number, we can extract the deck and the side of the boat. We could also extract distance to staircases, but this is effort-dense because we would need to look at Titanic maps

# In[8]:


def get_deck(cabin_name):
    deck = 564 
    if type(cabin_name) != float:
        deck = cabin_name[0]
    else:
        deck = 'Missing' #float('nan')
    return deck

def get_side(cabin_name):
    side = 564  
    if type(cabin_name) != float:
        n = cabin_name.split()
        nn = n[0]
        if len(nn) > 1:
            num = int((nn[1:]))
            if num % 2 == 0:
                side = 1 #'right'
            else:
                side = 2 #'left'
        else:
            side = 0 #'Missing' #float('nan') 
    else:
        side = 0 #'Missing' #float('nan') 
    return side
    
def add_deck_and_side(frame):
    Deck = []
    Side = []
    Cabin_missing = []
    for cabin_item in frame["Cabin"]:
        deck = get_deck(cabin_item)
        if deck == 'Missing':
            Cabin_missing.append(1)
        else:
            Cabin_missing.append(0)   
        side = get_side(cabin_item)
        Deck.append(deck)
        Side.append(side)
    np.array(Deck)
    np.array(Side)
    frame["Deck"] = Deck
    frame["Side"] = Side
    frame["Cabin_missing"] = Cabin_missing
    frame.drop(['Cabin'], axis=1)
    frame.head()
    return frame


# In[9]:


for frame in all_xframes:
    frame = add_deck_and_side(frame)
Xdata_all.head()


# Researching the new Deck and Side values, to see how they relate to the survival rate:

# In[10]:


print (Xdata_trainval_withanswers[["Deck", "Survived"]].groupby(['Deck'], as_index=False).mean())


# In[11]:


print (Xdata_trainval_withanswers[["Side", "Survived"]].groupby(['Side'], as_index=False).mean())


# So these values have some potential.
# 
# Suggestion from https://www.quora.com/How-do-I-deal-with-a-lot-more-than-half-of-missing-value:
# "I recommend replacing the missing values in a given column with a constant value and creating an additional indicator variable that encodes when a value is missing." 
# 
# So in my functions get_deck and get_side, I am adding the 'Unknown' value, and adding also another feature "Cabin_missing"

# In[12]:


print('Columns with null values:\n', Xdata_all.isnull().sum())
print("-"*10)


# ### Missing value 4: Embarked.
# #### Replace by most frequent value

# In[13]:


Xdata_all["Embarked"].value_counts()


# In[14]:


for frame in all_xframes:
    frame["Embarked"] = frame["Embarked"].fillna('S')
    frame['Embarked'] = frame['Embarked'].map( {'S': 1, 'C': 2, 'Q': 3} ).astype(int)

print('Columns with null values:\n', Xdata_all.isnull().sum())
print("-"*10)


# In[15]:


Xdata_all.head(10)


# ## Add more features
# 
# ### a) Add FamilySize and IsAlone features

# In[16]:


for frame in all_xframes:
    frame["FamilySize"] = np.add(frame['SibSp'], frame['Parch'])
    IsAlone = []
    for Relatives in frame["FamilySize"]:
        if Relatives == 0:
            IsAlone.append(1)
        else:
            IsAlone.append(0)
    frame['IsAlone']  = np.array(IsAlone) 
    
Xdata_all.head()


# ### b) Extracting titles out of participant names
# (Idea I took from multiple Kernels)

# In[17]:


def get_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return

for frame in all_xframes:
    frame['Title'] = frame['Name'].apply(get_title)
    frame['Title'] = frame['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    frame['Title'] = frame['Title'].replace('Mlle', 'Miss')
    frame['Title'] = frame['Title'].replace('Ms', 'Miss')
    frame['Title'] = frame['Title'].replace('Mme', 'Mrs')    

#print(pd.crosstab(Xdata_all['Title'], Xdata_all['Sex']))
print(Xdata_all.head())


# This is all features I can think of. 
# 
# Next step - normalizing features and bringing them to categorical form etc.
# 
# ## Normalizing features

# In[18]:


# checking variance - looking if any features need normalizing
Xdata_all.var().round(3)


# Want to normalize the Fare column.

# In[19]:


# Specify which column to normalize

# for some magical reason, iterarion "for frame in all_xframes" doesn't work.
# So I need to normalize data in each xframe individually.

# at least let's define a function

def norm_frame(frame, column):
    frame=frame.replace({column: {0: 0.1}})
    frame['Fare_norm'] = frame[column].apply(np.log)
    return frame

Xdata_all = norm_frame(Xdata_all, 'Fare')
Xdata_trainval = norm_frame(Xdata_trainval, 'Fare')
Xdata_test = norm_frame(Xdata_test, 'Fare')
Xdata_trainval_withanswers = norm_frame(Xdata_trainval_withanswers, 'Fare')

all_xframes = [Xdata_all, Xdata_trainval, Xdata_test, Xdata_trainval_withanswers]

# Result:
print(round(Xdata_all.var(),3))


# ## Age - put values into bins
# (Idea taken from Kernels)

# In[20]:


# Mapping Age
for dataset in all_xframes:
    dataset.loc[ dataset['Age'] <= 5, 'Age'] = int(1)                           # toddler
    dataset.loc[(dataset['Age'] > 5) & (dataset['Age'] <= 15), 'Age'] =  int(2)  # child & teen
    dataset.loc[(dataset['Age'] > 15) & (dataset['Age'] <= 30), 'Age'] = int(3)  # young adult
    dataset.loc[(dataset['Age'] > 30) & (dataset['Age'] <= 50), 'Age'] = int(4)  # adult
    dataset.loc[(dataset['Age'] > 50) & (dataset['Age'] <= 65), 'Age'] = int(5)  # mature
    dataset.loc[ dataset['Age'] > 65, 'Age'] = int(6)                            # old
    
print(Xdata_all.head())


# ## Change all features to categorical form

# In[21]:


title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
sex_mapping = {"male": 1, "female": 2}
deck_mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7, 'T': 8, 'Missing': 0}

for frame in all_xframes:
    frame['Sex'] = frame['Sex'].map(sex_mapping).astype(int)
    frame['Deck'] = frame['Deck'].map(deck_mapping).astype(int)
    frame['Title'] = frame['Title'].map(title_mapping).astype(int)
    frame['Title'] = frame['Title'].fillna(0)
#Xdata_all.head()


# In[22]:


Xdata_trainval.head()


# ## Drop the columns we can't use anymore

# In[23]:


for frame in all_xframes:
    frame.drop(["Name","Ticket","Cabin"], axis=1, inplace=True)
Xdata_all.head()


# #### Neat!
# # EXPLORATORY DATA ANALYSIS -
# ## Plotting correlations between features

# In[ ]:


#correlation heatmap of dataset
def correlation_heatmap(df):
    _ , ax = plt.subplots(figsize =(14, 12))
    colormap = sns.diverging_palette(220, 10, as_cmap = True)
    
    _ = sns.heatmap(
        df.corr(), 
        cmap = colormap,
        square=True, 
        cbar_kws={'shrink':.9 }, 
        ax=ax,
        annot=True, 
        linewidths=0.1,vmax=1.0, linecolor='white',
        annot_kws={'fontsize':12 }
    )
    
    plt.title('Pearson Correlation of Features', y=1.05, size=15)

correlation_heatmap(Xdata_all)


# We could eliminate some of the features Cabin_missing, Side, Deck. They are correlated because they are all derived from the same variable. It is to be exptected.
# 
# But for now, we will keep all the features.
# 

# # CLASSIFICATION!
# ## Import all the classifiers, 
# ## define all the functions
# 
# 

# In[ ]:


from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from tpot import TPOTClassifier
from sklearn.metrics import roc_auc_score

classifiers = [
    KNeighborsClassifier(3),
    SVC(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis(),
    LogisticRegression(),
    MLPClassifier(),
    LinearSVC(),
    XGBClassifier(),
    Perceptron()]

def compare_with_tpot(data_train_current,data_train_answers,data_test_current,data_test_answers):
    tpot = TPOTClassifier(
        generations=5,
        population_size=20,
        verbosity=2,
        scoring='accuracy',
        random_state=42,
        disable_update_check=True,
        config_dict='TPOT light')
    tpot.fit(data_train_current, data_train_answers)
    print ("TPOT results:")
    print (tpot.score(data_test_current, data_test_answers))
    print('TPOT results are estimated with train data')

def compare_classifiers(classifiers,data_train_current,data_train_answers,data_test_current,data_test_answers):
    # for plots
    acc_dict = {}
    log_cols = ["Classifier", "Accuracy"]
    log  = pd.DataFrame(columns=log_cols)

    acc_dict_train = {}
    log_train  = pd.DataFrame(columns=log_cols)
    # for table
    tab_dict = {}
    for clf in classifiers:
        name = clf.__class__.__name__
        clf.fit(data_train_current, data_train_answers)
        acc_train = clf.score(data_train_current, data_train_answers)
        acc_test = clf.score(data_test_current, data_test_answers)
        #update dict for plot
        if name in acc_dict:
            acc_dict[name] += acc_test
        else:
            acc_dict[name] = acc_test
        if name in acc_dict_train:
            acc_dict_train[name] += acc_train
        else:
            acc_dict_train[name] = acc_train        
        #update dict for table
        tab_dict[name] = [round(acc_train,4), round(acc_test,4)]
    # table
    table = pd.DataFrame.from_dict(tab_dict,orient='index',columns=['On training set', 'On validation set'])
    table.sort_values('On training set', axis=0, ascending=False, inplace=True)
    print("\n Sorted by performance on training set:")
    print(table)
    table.sort_values('On validation set', axis=0, ascending=False, inplace=True)
    print("\n Sorted by performance on validation set:")
    print(table)

    
"""
# Code to use for plotting the classifier results if needed:
for clf in acc_dict:
    acc_dict_train[clf] = acc_dict_train[clf] / 10.0
    log_entry_train = pd.DataFrame([[clf, acc_dict_train[clf]]], columns=log_cols)
    log_train = log_train.append(log_entry_train)
plt.xlabel('Accuracy')
plt.title('Classifier Accuracy on training set')
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log_train, color="r")"""


# In[ ]:


def calculate_score(clf, x_train, x_test, y_train, y_test):
    n = 10
    accuracies = []
    while n > 0:
        clf.fit(x_train, y_train)
        prediction = clf.predict(x_test)
        accuracies.append((sklearn.metrics.accuracy_score(y_test, prediction)))
        n -= 1
    av_acc = sum(accuracies)/float(len(accuracies))
    return av_acc

def write_submission(prediction, submission_name, data_test_answers=data_test_answers):
    output = data_test_answers.copy(deep=True)
    output["Survived"] = prediction
    output.to_csv(submission_name)
    
def make_prediction_and_write_submission(submission_name, clf, x_train, y_train, X_test, data_test_answers=data_test_answers):
    clf = clf.fit(x_train,y_train)
    prediction = clf.predict(X_test)
    write_submission(prediction, submission_name, data_test_answers)

def select_features_and_split(xdata, ydata, 
                              feature_list=['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked','Deck','Side','Cabin_missing','FamilySize','IsAlone','Title','Fare_norm']):
    xdata_features = xdata[feature_list]
    xdata_train, xdata_val = train_test_split(xdata_features, test_size=0.2)
    Ydata_train, Ydata_val = train_test_split(ydata, test_size=0.2)
    ydata_train = np.ravel(Ydata_train)
    ydata_val = np.ravel(Ydata_val)
    return (xdata_train, xdata_val, ydata_train, ydata_val)


# ## Baseline: All the given features

# In[ ]:


features=['Pclass','Sex','Age','SibSp','Parch','Fare'] # given features
xdata_train, xdata_val, ydata_train, ydata_val = select_features_and_split(Xdata_trainval, Ydata_trainval, features)
print("Given bl features:")
compare_classifiers(classifiers, xdata_train, ydata_train, xdata_val, ydata_val)
compare_with_tpot(xdata_train, ydata_train, xdata_val, ydata_val)


# ## (1) All the features we added 

# In[ ]:


features=['Pclass','Sex','Age','SibSp','Parch','Embarked','Deck','Side','Cabin_missing','FamilySize','IsAlone','Title','Fare_norm'] # all features
xdata_train, xdata_val, ydata_train, ydata_val = select_features_and_split(Xdata_trainval, Ydata_trainval, features)
print("All possible features:")
compare_classifiers(classifiers, xdata_train, ydata_train, xdata_val, ydata_val)
compare_with_tpot(xdata_train, ydata_train, xdata_val, ydata_val)


# #### Submit best results as our baseline         

# In[ ]:


# On validation set (+ Kaggle scores):
# 0.77033 LogisticRegression                      0.6110             0.6480
make_prediction_and_write_submission('mybaseline_190401_logreg.csv', LogisticRegression(), Xdata_trainval, Ydata_trainval, Xdata_test, data_test_answers)
# 0.71770 LinearSVC                               0.6053             0.6480
make_prediction_and_write_submission('mybaseline_190401_linearSVC.csv', LinearSVC(), Xdata_trainval, Ydata_trainval, Xdata_test, data_test_answers)
# 0.77511 LinearDiscriminantAnalysis              0.6110             0.6425
make_prediction_and_write_submission('mybaseline_190401_LDA.csv', LinearDiscriminantAnalysis(), Xdata_trainval, Ydata_trainval, Xdata_test, data_test_answers)
# 0.76076 MLPClassifier                           0.6124             0.6369
make_prediction_and_write_submission('mybaseline_190401_MLPC.csv', MLPClassifier(), Xdata_trainval, Ydata_trainval, Xdata_test, data_test_answers)
# 0.71770 SVC                                     0.6798             0.5978
make_prediction_and_write_submission('mybaseline_190401_SVC.csv', SVC(), Xdata_trainval, Ydata_trainval, Xdata_test, data_test_answers)

# On training set (+ Kaggle scores):
# 0.75598 DecisionTreeClassifier                  0.8834             0.5642
make_prediction_and_write_submission('mybaseline_190401_DT.csv', DecisionTreeClassifier(), Xdata_trainval, Ydata_trainval, Xdata_test, data_test_answers)
# 0.77511 RandomForestClassifier                  0.8694             0.5307
make_prediction_and_write_submission('mybaseline_190401_RanFor.csv', RandomForestClassifier(), Xdata_trainval, Ydata_trainval, Xdata_test, data_test_answers)
# 0.77511 GradientBoostingClassifier              0.7430             0.5642
make_prediction_and_write_submission('mybaseline_190401_GBoost.csv', GradientBoostingClassifier(), Xdata_trainval, Ydata_trainval, Xdata_test, data_test_answers)
# 0.77033 XGBClassifier                           0.7149             0.5419
make_prediction_and_write_submission('mybaseline_190401_XGBC.csv', XGBClassifier(), Xdata_trainval, Ydata_trainval, Xdata_test, data_test_answers)
# 0.68899 KNeighborsClassifier                    0.7107             0.5531
make_prediction_and_write_submission('mybaseline_190401_KNN.csv', KNeighborsClassifier(), Xdata_trainval, Ydata_trainval, Xdata_test, data_test_answers)


# #### Best scores on Kaggle leaderoard:
# ##### 0.77511 LinearDiscriminantAnalysis    
# ##### 0.77511 RandomForestClassifier       
# ##### 0.77511 GradientBoostingClassifier   # very similar to RandomForestClassifier 
# ##### 0.77033 LogisticRegression                  
# ##### 0.77033 XGBClassifier                        

# About whether to do feature selection or hyperparameter tuning first - https://stats.stackexchange.com/questions/264533/how-should-feature-selection-and-hyperparameter-optimization-be-ordered-in-the-m
# 
# In short:
# - If can afford it, do it at the same time. All feature combinations and all (reasonable) model parameters.
# - Else, 
#     + (1) Set the model with ok-ish params, 
#     + (2) Select features, 
#     + (3) Fine-tune the params
#     

# # APPROACH 1 - DUMB AND LONG
# # ITERATE THROUGH ALL FEATURE SUBSETS AND ALL POSSIBLE PARAMS OF BEST PERFORMING CLASSIFIERS
# 

# In[ ]:


def tuning_model(model_name, data_train, data_test, y_train, y_test):
    mydict = {'LDA_with_shrinkage'  : {'model' : LinearDiscriminantAnalysis(),
                                       'params': {"solver"          : ["lsqr"],
                                                  "shrinkage"       : [None, "auto", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                                                  "priors"          : [None],
                                                  "tol"             : [0.0001, 0.001, 0.01, 0.1, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
                                                   }
                                      },
             'LDA_without_shrinkage':  {'model' : LinearDiscriminantAnalysis(), 
                                        'params': {"solver"          : ["svd"],
                                                   "shrinkage"       : [None],
                                                   "priors"          : [None],
                                                   "store_covariance": [False, True],
                                                   "tol"             : [0.0001, 0.001, 0.01, 0.1, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
                                                  }
                                       },
                               'SVM':  {'model' : LinearSVC(), 
                                        'params': {"penalty"          : ["l2","l1"], 
                                                  "loss"             : ["squared_hinge"], # hinge gives an error when dual = False
                                                  "dual"             : [False], # False because n_samples >> n_features.
                                                  "C"                : [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
                                                  "multi_class"      : ["ovr", "crammer_singer"], 
                                                  "fit_intercept"    : [False, True],
                                                  "intercept_scaling": [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
                                                  "class_weight"     : [None],
                                                  "random_state"     : [None], 
                                                  "max_iter"         : [10000], 
                                                  "tol"              : [0.0001, 0.001, 0.01, 0.1, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
                                                 }
                                       },
                     'LogReg_solver1': {'model' : LogisticRegression(), 
                                        'params': {"penalty"         : ["l1", "l2"],
                                                  "dual"             : [False],
                                                  "C"                : [0.001, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
                                                  "fit_intercept"    : [False, True],
                                                  "intercept_scaling":[0.001, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
                                                  "class_weight"     : [None],
                                                  "random_state"     : [None], 
                                                  "solver"           : ["liblinear", "saga"], 
                                                  "max_iter"         : [1000], 
                                                  "multi_class"      : ["auto"], 
                                                  "verbose"          :[0], 
                                                  "warm_start"       :[False, True], 
                                                  "n_jobs"           :[None],
                                                  "tol"              : [0.0001, 0.001, 0.01, 0.1, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
                                                 }
                                       }, 
                    'LogReg_solver2': {'model' : LogisticRegression(), 
                                       'params': {"penalty"          : ["l2"],
                                                  "dual"             : [False],
                                                  "C"                : [0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
                                                  "fit_intercept"    : [False, True],
                                                  "intercept_scaling":[0.001, 0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
                                                  "class_weight"     : [None],
                                                  "random_state"     : [None], 
                                                  "solver"           : ["newton-cg", "lbfgs", "sag"], 
                                                  "max_iter"         : [1000], 
                                                  "multi_class"      : ["auto"], 
                                                  "verbose"          :[0], 
                                                  "warm_start"       :[False, True], 
                                                  "n_jobs"           :[None],
                                                  "tol"              : [0.0001, 0.001, 0.01, 0.1, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009]
                                                 }
                                       }, 
                      'RandomForest': {'model' : RandomForestClassifier(),
                                       'params': {"n_estimators"            : [2,5,10,20,50,100,200],
                                                  "criterion"               : ['gini','entropy'],
                                                  "max_depth"               : [None, 1,2,5,10,20,100],
                                                  "min_samples_split"       : [0.01,0.05,0.1,0.5,5,10],
                                                  "min_samples_leaf"        : [1,2,5,10],
                                                  "min_weight_fraction_leaf": [0.0,0.01,0.05,0.1],
                                                  "max_features"            : [None, 'auto', 'log2'],
                                                  "max_leaf_nodes"          : [None],
                                                  "min_impurity_decrease"   : [0.0],
                                                  "bootstrap"               : [True, False]
                                                }
                                      },
                     'XGBClassifier': {'model' : XGBClassifier(),
                                       'params':{"max_depth" : [2,3,5,10],
                                                 "learning_rate": [0.001,0.003,0.005,0.007,0.01,0.02,0.05],
                                                 "n_estimators": [10,20,50,100,200,500],
                                                 "objective": ['binary:logistic'],
                                                 "booster": ["gbtree", "gblinear", "dart"],
                                                 "min_child_weight":[0.1,0.5,1.0],
                                                 "reg_alpha": [0.0,0.1,0.5,0.7,1.0],
                                                 "reg_lambda": [0.0,0.1,0.5,0.7,1.0]

                                       }
                         
                     }
             }


    clf = mydict[model_name]['model']
    parameters = mydict[model_name]['params']
    grid_cv = GridSearchCV(clf, parameters, scoring = make_scorer(accuracy_score))
    grid_cv = grid_cv.fit(data_train, y_train)    
    #print("Our optimized LDA model is:")
    #print(grid_cv.best_estimator_ )
    #print("Score: ")    
    #print(calculate_score(grid_cv.best_estimator_, data_train_normed, data_test_normed, data_train_answers, data_test_answers))
    score = sklearn.metrics.accuracy_score(y_test, grid_cv.best_estimator_.predict(data_test))
    #score = calculate_score(grid_cv.best_estimator_, data_train, data_test, y_train, y_test)
    return (grid_cv.best_estimator_, score)

"""best_model, score = tuning_model('RandomForest', xdata_train, xdata_val, ydata_train, ydata_val)
print(best_model)
print(score)
print(str(best_model))"""


# # Next section takes VERY LONG to run
# # Run at your own risk!

# In[ ]:


"""# Function to iterate through all possible subsets of features
def iterate_through_features(ss):
    return chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1)))

features=['Pclass','Sex','Age','SibSp','Parch','Embarked','Deck','Side','Cabin_missing','FamilySize','IsAlone','Title','Fare_norm'] # all features
best_classifiers=['LDA_with_shrinkage','LDA_without_shrinkage','SVM','LogReg_solver1','LogReg_solver2','RandomForest','XGBClassifier']
#best_classifiers=['LDA_with_shrinkage','LDA_without_shrinkage','SVM','LogReg_solver1','LogReg_solver2','XGBClassifier']

best_result = {'score':0,
              'model':'',
              'features':[]}

for feature_subset in iterate_through_features(features):
    if len(feature_subset) > 0:
        myfeats = list(feature_subset)
        print(myfeats)
        xdata_train, xdata_val, ydata_train, ydata_val = select_features_and_split(Xdata_trainval, Ydata_trainval, myfeats)
        for myclf in best_classifiers:
            print('\n\n')
            print(myclf)
            (model, currentscore) = tuning_model(myclf, xdata_train, xdata_val, ydata_train, ydata_val)
            print(model)
            print('Current score:')
            print(currentscore)
            if currentscore > best_result['score']:
                best_result['score'] = currentscore
                best_result['model'] = model
                best_result['features'] = myfeats
                print('Best score:')
                print(best_result)
                
print("AND THE ABSOLUTE CHAMPION IS")
print(best_result)
            """


# In[ ]:


#print(best_result)


# # A MUCH MORE REALISTIC APPROACH...

# In[ ]:


top_classifiers = [
    LinearDiscriminantAnalysis(),
    LinearSVC(),
    LogisticRegression(),
    RandomForestClassifier(),
    GradientBoostingClassifier(),
    XGBClassifier(),
    KNeighborsClassifier(),
    GaussianNB(),
    MLPClassifier()]

features=['Pclass','Sex','Age','SibSp','Parch','Embarked','Deck','Side','Cabin_missing','FamilySize','IsAlone','Title','Fare_norm'] # all features
Xdata_train, Xdata_val, ydata_train, ydata_val = select_features_and_split(Xdata_trainval, Ydata_trainval, features)
Ydata_train = pd.DataFrame(data=ydata_train, index=range(1,len(ydata_train)+1), columns=['Survived'])
Ydata_val = pd.DataFrame(data=ydata_val, index=range(1,len(ydata_val)+1), columns=['Survived'])

feature_selection = [
    RFE(LinearSVC(), n_features_to_select=None, step=1, verbose=0),
    RFE(DecisionTreeClassifier(), n_features_to_select=None, step=1, verbose=0),  
    SelectFromModel(LinearSVC())]


best_result = {'score':0,
              'model':'',
              'featalg':'',
              'features':[]}

for my_clf in top_classifiers:
    for my_feat in feature_selection:
        clf = Pipeline([
            ('feature_selection', my_feat),
            ('classification', my_clf)
        ])
        clf.fit(Xdata_train, Ydata_train)
        acc = clf.score(Xdata_val, Ydata_val)
        print("\nClassifier:")
        print(my_clf)
        print("Feature selection:")
        print(my_feat)
        print("Accuracy:")
        print(acc)    
        if acc > best_result['score']:
            best_result['score'] = acc
            best_result['model'] = my_clf
            best_result['featalg'] = my_feat
            f = my_feat.fit(Xdata_train, Ydata_train)
            f1 = list(Xdata_train.columns.values.tolist())
            f2 = f.support_
            flist = []
            for i, val in enumerate(f1):
                print(val, f2[i])
                if f2[i] == True:
                    flist.append(val)        
            best_result['features'] = flist

print("\n\n\nAND THE CHAMPION IS")
print(best_result)
            


# In[ ]:


# Best pipeline: 
# 'model': LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
#            solver='svd', store_covariance=False, tol=0.0001), 
# 'featalg': RFE(estimator=DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
#            max_features=None, max_leaf_nodes=None,
#            min_impurity_decrease=0.0, min_impurity_split=None,
#            min_samples_leaf=1, min_samples_split=2,
#            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
#            splitter='best')


# a) implementing DTC feature selection
# features algorithm chose are:
feature_list = best_result['features']
data_trainval = Xdata_trainval[feature_list]
data_test = Xdata_test[feature_list]

clf = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None, solver='svd', store_covariance=False, tol=0.000)
make_prediction_and_write_submission('190402_bestpipeline_LDA.csv', clf, data_trainval, Ydata_trainval, data_test, data_test_answers)

clf = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=3, min_child_weight=1, missing=None, n_estimators=100,
       n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)
make_prediction_and_write_submission('190402_bestpipeline_XGBC.csv', clf, data_trainval, Ydata_trainval, data_test, data_test_answers)

print("Done")


# Kaggle score - 
# LDA 0.70334
# XGBC 0.78947
# 

# ## Hyper-tuning parameters of XGBC

# In[ ]:


data_train = Xdata_train[feature_list]
data_val = Xdata_val[feature_list]
data_test = Xdata_test[feature_list]

clf_XGBC_tuned, score = tuning_model('XGBClassifier', data_train, data_val, Ydata_train, Ydata_val)
print(clf_XGBC_tuned)
print(score)

make_prediction_and_write_submission('190402_bestpipeline_XGBC_hypertuned.csv', clf_XGBC_tuned, data_trainval, Ydata_trainval, data_test, data_test_answers)

print("Done")


# In[ ]:





# In[ ]:




