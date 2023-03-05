# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

#!sudo apt install docker -y

#!pip install numpy pandas boto3 matplotlib scikit-learn 

#pip install lazypredict

#!pip install plotly

#!pip install mlflow

#!pip install numba
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import lazypredict
from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import mlflow
import mlflow.sklearn
import itertools
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn import tree
from mlflow.models.signature import infer_signature
#!pip install kaggle

mlflow.set_experiment('my_adidas_model5')

df=pd.read_csv('/home/ubuntu/adidas/kaggle/input/spaceship-titanic/train.csv')
df.head()


df.shape

df.info()

df.dtypes

px.histogram(df,x='HomePlanet',color='Transported',barmode='group')

px.histogram(df,x='CryoSleep',color='Transported',barmode='group')

px.histogram(df,x='Destination',color='Transported',barmode='group')

px.histogram(df,x='VIP',color='Transported',barmode='group')

#cols=df.select_dtypes('object').columns.tolist()
#for i in cols:
#    ct=df[i].value_counts()
 #   plt.title(i);
 #   ct.plot(kind='bar')
 #   plt.figure(figsize=(8,5));
    
 #   plt.show();

cols=df.select_dtypes('object').columns
cols.tolist()

df['HomePlanet'].fillna(df['HomePlanet'].value_counts().index[0],inplace=True)


def missingvalue(df):
    cols=df.select_dtypes('object').columns
    cols=cols.tolist()
    #for i in cols:
    df['HomePlanet'].fillna(df['HomePlanet'].value_counts().index[0],inplace=True)
    df['CryoSleep'].fillna(df['CryoSleep'].value_counts().index[0],inplace=True)
    df['Destination'].fillna(df['Destination'].value_counts().index[0],inplace=True)
    df['VIP'].fillna(df['VIP'].value_counts().index[0],inplace=True)


    cols1=df.select_dtypes('float64').columns
    cols1=cols1.tolist()
    for i in cols1:
        df[i]=df[i].fillna(df[i].mean())
    return df

def Onehotencoding(df1):
    df1=df1.join(pd.get_dummies(df['HomePlanet'],prefix='HomePlanet',prefix_sep='_'))
    df1=df1.join(pd.get_dummies(df['CryoSleep'],prefix='CryoSleep',prefix_sep='_'))
    df1=df1.join(pd.get_dummies(df['Destination'],prefix='Destination',prefix_sep='_'))
    df1=df1.join(pd.get_dummies(df['VIP'],prefix='VIP',prefix_sep='_'))
    df1.drop(['HomePlanet','CryoSleep','Destination','VIP'],axis=1,inplace=True)
    return df1

    

def pre_processing(df):
    df.drop(['PassengerId','Name','Cabin'],axis=1,inplace=True)
    df=missingvalue(df)
    #df1=df.apply(lambda x:x.fillna(x.value_counts().index[0]))
    #print(df1.info())
    cols=df.select_dtypes('object').columns.tolist()
    df=Onehotencoding(df)
    #for i in cols:
    #    df1=df.join(pd.get_dummies(df[i],prefix=i,prefix_sep='_'))
    #df1.drop(cols,axis=1,inplace=True)
    #print(df1.info())
    #scaler=StandardScaler()
    #scaled=scaler.fit_transform(df1)
    #df2=pd.DataFrame(scaled,index=df1.index,columns=df1.columns)
    return df
    

#def transform1(df):
#    scaler=StandardScaler()
#    scaled=scaler.fit_transform(df)
#    df1=pd.DataFrame(scaled,index=df.index,columns=df.columns)
#    return df1

df1=pre_processing(df)

#df1

#df1.info()

y=df1['Transported']
col=df1.columns

col=col.delete(6)
x=df1[col]
#x=transform1(x)

X_train,X_test,y_train,y_test=train_test_split(x,y,random_state=5,stratify = y,test_size = 0.40)

with mlflow.start_run(run_name='My model experiment') as run:

        clf=LazyClassifier()
        model,predictions=clf.fit(X_train,X_test,y_train,y_test)

        print(model)

        ### LGBMClassifier has better performance compared to other classifiers

        clf=lgb.LGBMClassifier(random_state=5)
        clf.fit(X_train,y_train)
        pred=clf.predict(X_test)

        clf.get_params()

        #acc=accuracy_score(y_test,pred)
        #acc
        print(clf.score(X_train,y_train))
        print(clf.score(X_test,y_test))

        ### Check if the model does not overfit

        y_pred_train=clf.predict(X_train)

        print('{0:0.4f}'.format(accuracy_score(y_train,y_pred_train)))

        ### The accuracy of train and test set are comparable**

        ### Hyperparameter tuning
        """
        param_grid={'max_bin':[150,250],'learning_rate':[0.13,0.03],'num_iterations':[150,300],'min_gain_to_split':[0.1,1],'max_depth':[10,20]}
        clf=RandomizedSearchCV(estimator=clf,param_distributions=param_grid, scoring='accuracy')
        num_iterations = 150
        min_gain_to_split= 1
        max_depth=20
        max_bin = 150
        learning_rate= 0.13
        mlflow.log_param('num_iterations',num_iterations)
        mlflow.log_param('min_gain_to_split',min_gain_to_split)
        mlflow.log_param('learning_rate',learning_rate)
        mlflow.log_param('max_depth',max_depth)
        mlflow.log_param('max_bin',max_bin)
       
        search=clf.fit(X_train,y_train)
        search.best_params_

        search.best_score_
        """
        clf=lgb.LGBMClassifier(max_bin=250,learning_rate=0.03,num_iterations=150,min_gain_to_split=1,max_depth=20)
        clf.fit(X_train,y_train)
        pred=clf.predict(X_test)
        #acc=accuracy_score(y_test,pred)
        #acc
        print("Accuracy on train data",clf.score(X_train,y_train))
        print("Accuracy on test data",clf.score(X_test,y_test))

        clf=lgb.LGBMClassifier(max_bin=250,learning_rate=0.13,num_iterations=150,min_gain_to_split=0.3,max_depth=20)
        clf.fit(X_train,y_train)
        pred=clf.predict(X_test)
        #acc=accuracy_score(y_test,pred)
        #print("accuracy",acc)
        print("Accuracy on train data",clf.score(X_train,y_train))
        print("Accuracy on test data",clf.score(X_test,y_test))
        
        mlflow.sklearn.log_model(clf, 'random-forest-model')
    
from sklearn.metrics import precision_score, recall_score, f1_score
        
def model_feature_importance(model):
    feature_importance = pd.DataFrame(
        model.feature_importances_,
        index=X_train.columns,
        columns=["Importance"],
    )

    # sort by importance
    feature_importance.sort_values(by="Importance", ascending=False, inplace=True)

    # plot
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=feature_importance.reset_index(),
        y="index",
        x="Importance",
    ).set_title("Feature Importance")
    # save image
    plt.savefig("model_artifacts/feature_importance.png", bbox_inches='tight')
    
# Track params and metrics
def log_mlflow_run(model, signature):
    # Auto-logging for scikit-learn estimators
    # mlflow.sklearn.autolog()

    # log estimator_name name
    name = model.__class__.__name__
    mlflow.set_tag("estimator_name", name)

    # log input features
    mlflow.set_tag("features", str(X_train.columns.values.tolist()))

    # Log tracked parameters only
    mlflow.log_params({key: model.get_params()[key] for key in param_grid})

    #mlflow.log_metrics({
     #   'RMSE_CV': score_cv.mean(),
     #   'RMSE': score,
    #})

    # log training loss
    #for s in model.train_score_:
     #   mlflow.log_metric("Train Loss", s)

    # Save model to artifacts
    mlflow.sklearn.log_model(model, "model", signature=signature)

    # log charts
    mlflow.log_artifacts("model_artifacts")    

def model_permutation_importance(model):
    p_importance = permutation_importance(model, X_test, y_test, random_state=42, n_jobs=-1)

    # sort by importance
    sorted_idx = p_importance.importances_mean.argsort()[::-1]
    p_importance = pd.DataFrame(
        data=p_importance.importances[sorted_idx].T,
        columns=X_train.columns[sorted_idx]
    )

    # plot
    plt.figure(figsize=(12, 8))
    sns.barplot(
        data=p_importance,
        orient="h"
    ).set_title("Permutation Importance")

    # save image
    plt.savefig("model_artifacts/permutation_importance.png", bbox_inches="tight")
"""
def model_tree_visualization(model):
    # generate visualization
    tree_dot_data = tree.export_graphviz(
        #decision_tree=model.n_estimators,  # Get the first tree,
        label="all",
        feature_names=X_train.columns,
        filled=True,
        rounded=True,
        proportion=True,
        impurity=False,
        precision=1,
    )

    # save image
    graph_from_dot_data(tree_dot_data).write_png("model_artifacts/Decision_Tree_Visualization.png")

    # show tree
    return graphviz.Source(tree_dot_data)

    
        # log model performance 
        #mse = mean_squared_error(y_test, pred)
        prc = precision_score(y_test, pred)
        mlflow.log_metric('precision', prc)
        print('precision_score: %f' % prc)
        
        rs = recall_score(y_test, pred)
        mlflow.log_metric('Recall', rs)
        print('Recall_score: %f' % rs)
        
        f1 = f1_score(y_test, pred)
        mlflow.log_metric('f1_score', f1)
        print('f1_score: %f' % f1)
        
        """
model_class = lgb.LGBMClassifier
param_grid={
    'max_bin':[150,250],
    'learning_rate':[0.13,0.03],
    'num_iterations':[150,300],
    'min_gain_to_split':[0.1,1],
    'max_depth':[10,20]
}
        
# generate parameters combinations
params_keys = param_grid.keys()
params_values = [
    param_grid[key] if isinstance(param_grid[key], list) else [param_grid[key]]
    for key in params_keys
]
runs_parameters = [
    dict(zip(params_keys, combination)) for combination in itertools.product(*params_values)
]

# training loop
for i, run_parameters in enumerate(runs_parameters):
    print(f"Run {i}: {run_parameters}")

    # mlflow: stop active runs if any
    if mlflow.active_run():
        mlflow.end_run()
    # mlflow:track run
    mlflow.start_run(run_name=f"Run {i}")

    # create model instance
    model = model_class(**run_parameters)
       

    # train
    model.fit(X_train, y_train)

    # get evaluations scores
    #score = rmse_score(y_test, model.predict(X_test))
    #score_cv = rmse_cv_score(model, X_train, y_train)
    #prcision = precision_score(y_test, model.predict(X_test))
    #f1score = f1_score(y_test, model.predict(X_test))
    #recall = recall_score(y_test, model.predict(X_test))
    
    prc = precision_score(y_test, pred)
    mlflow.log_metric('precision', prc)
    print('precision_score: %f' % prc)
    
    rs = recall_score(y_test, pred)
    mlflow.log_metric('Recall', rs)
    print('Recall_score: %f' % rs)
    
    f1 = f1_score(y_test, pred)
    mlflow.log_metric('f1_score', f1)
    print('f1_score: %f' % f1)
    
    
    # generate charts
    model_feature_importance(model)
    plt.close()
    model_permutation_importance(model)
    plt.close()
    #model_tree_visualization(model)

    # get model signature
    signature = infer_signature(model_input=X_train, model_output=model.predict(X_train))

    # mlflow: log metrics
    log_mlflow_run(model, signature)

    # mlflow: end tracking
    mlflow.end_run()
    print("")

