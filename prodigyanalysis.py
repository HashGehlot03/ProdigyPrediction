import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer,IterativeImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import *
import pickle




data = pd.read_csv('ProdigyTrain.csv')
data_x = data.drop(['employee_id','is_promoted','region'],axis = 1)  #Independent variables
data_x.rename(columns = {})
data_y = data['is_promoted'] #dependent variable
data_y.value_counts();       # not_promoted (0) is 32047 and promoted (1) is 2951
x_train,x_test,y_train,y_test = train_test_split(data_x,data_y,test_size = 0.2,stratify = data_y,shuffle = True) #labels are of different amount so we use stratify in train test splitting
x_train.reset_index(inplace = True,drop = True)
x_test.reset_index(inplace = True,drop = True)
num_cols = [col for col in data_x.columns if data_x[col].dtypes != 'O']   # ['no_of_trainings','age','previous_year_rating','length_of_service','KPIs_met >80%', 'awards_won?','avg_training_score']
cat_cols = [col for col in data_x.columns if data_x[col].dtypes == 'O']   # ['department', 'education', 'gender', 'recruitment_channel']
num_nan = [col for col in num_cols if data_x[col].isnull().any()]
cat_nan = [col for col in cat_cols if data_x[col].isnull().any()]
ordinal_feat = 'education'
nominal_feat = ['department','recruitment_channel']

def model_evaluation(x_train,y_train,x_test,preprocessing_pipe):
  #Prepare the copies of x_train and y_train for different model accuracy checking
  x_train_d,x_train_l,x_train_r,x_train_a,y_train_d,y_train_l,y_train_r,y_train_a,x_test_d,x_test_l,x_test_r,x_test_a = x_train.copy(),x_train.copy(),x_train.copy(),x_train.copy(),y_train.copy(),y_train.copy(),y_train.copy(),y_train.copy(),x_test.copy(),x_test.copy(),x_test.copy(),x_test.copy()
  #Different model evaluation checking
  decision_tree = make_pipeline(preprocessing_pipe,DecisionTreeClassifier())
  decision_tree.fit(x_train_d,y_train_d)
  pred_d = decision_tree.predict(x_test_d)
  print('(Decision Tree Classification)\n')
  print(f'On training set accuracy :- {cross_val_score(decision_tree,x_train_d,y_train_d,cv=8).mean()}')
  print(f'On test set accuracy :- {accuracy_score(y_test,pred_d)}')
  logreg = make_pipeline(preprocessing_pipe,LogisticRegression())
  logreg.fit(x_train_l,y_train_l)
  pred_l = logreg.predict(x_test_l)
  print('(Logistic Regression)\n')
  print(f'On training set accuracy :- {cross_val_score(logreg,x_train_l,y_train_l,cv=8).mean()}')
  print(f'On test set accuracy :- {accuracy_score(y_test,pred_l)}')
  rf = make_pipeline(preprocessing,DecisionTreeClassifier())
  rf.fit(x_train_r,y_train_r)
  pred_r = rf.predict(x_test_r)
  print('(Random Forest Classification)\n')
  print(f'On training set accuracy :- {cross_val_score(rf,x_train_r,y_train_r,cv=8).mean()}')
  print(f'On test set accuracy :- {accuracy_score(y_test,pred_r)}')
  ada = make_pipeline(preprocessing,DecisionTreeClassifier())
  ada.fit(x_train_a,y_train_a)
  pred_a = ada.predict(x_test_a)
  print('(Ada Boost Classification)\n')
  print(f'On training set accuracy :- {cross_val_score(ada,x_train_a,y_train_a,cv=8).mean()}')
  print(f'On test set accuracy :- {accuracy_score(y_test,pred_a)}')
#model_evaluation(x_train,y_train,x_test,preprocessing)
# Logistic Regression works the best


impute = make_column_transformer(
    (SimpleImputer(strategy = 'median'),[6]),
    (SimpleImputer(strategy = 'most_frequent'),[1]),
    remainder = 'passthrough'
)


encode = make_column_transformer(
    (OrdinalEncoder(categories = 'auto'),[1]),
    (OneHotEncoder(),[2,3,4]),
    remainder = 'passthrough'
)



params = {'logreg__C':[100, 10, 1.0, 0.1, 0.01],'logreg__penalty':['l1','l2'],'logreg__solver':['lbfgs', 'liblinear']}
preprocessing = make_pipeline(impute,encode)
pipe = Pipeline([('preprocessing',preprocessing),('logreg',LogisticRegression())])
main_pipe = GridSearchCV(pipe,params,cv=5,scoring = 'accuracy')
main_pipe.fit(x_train,y_train)
model_pipeline = main_pipe.best_estimator_
prodigy_model = pickle.dump(model_pipeline,open('ProdigyModel.sav','wb'))

mod = pickle.load(open('ProdigyModel.sav','rb'))
#mod.predict(x_test.iloc[0,:])

