from IPython.display import clear_output
import numpy    as np
import seaborn  as sb
import matplotlib.pyplot as plt
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.tree          import DecisionTreeClassifier
from sklearn.ensemble      import RandomForestClassifier
from sklearn.ensemble      import ExtraTreesClassifier
from sklearn.ensemble      import AdaBoostClassifier
from sklearn.ensemble      import GradientBoostingClassifier
from sklearn.experimental  import enable_hist_gradient_boosting # Necesary for HistGradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier


def data_to_pandas(pt):
    data=pd.read_csv(pt, index_col='PassengerId')
    return data
    
def column_to_trans(column):
    pass


def make_pipe(num_vars, cat_vars):
    num_4_treeModels = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean'))
    ])

    cat_4_treeModels= Pipeline(steps=[
    ('imputer,', SimpleImputer( strategy='most_frequent')),
    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])
  
    tree_prepro = ColumnTransformer(transformers=[
    ('num', num_4_treeModels, num_vars),
    ('cat', cat_4_treeModels, cat_vars),
    ], remainder='drop') 
    
    return tree_prepro


def model_types(num_vars,cat_vars):
    tree_classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=0),
    "Extra Trees":   ExtraTreesClassifier(random_state=0),
    "Random Forest": RandomForestClassifier(random_state=0),
    "AdaBoost":      AdaBoostClassifier(random_state=0),
    "Skl GBM":       GradientBoostingClassifier(random_state=0),
    "Skl HistGBM":   HistGradientBoostingClassifier(random_state=0),
    "XGBoost":       XGBClassifier(),
    "LightGBM":      LGBMClassifier(random_state=0),
  #"CatBoost":      CatBoostClassifier(n_estimators=100)
    }
    tree_classifiers = {name: make_pipeline(make_pipe(num_vars, cat_vars), model) for name, model in tree_classifiers.items()}
    return tree_classifiers