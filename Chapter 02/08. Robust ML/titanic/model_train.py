from unittest import result
import data_compose as dc
import pandas as pd
from sklearn.metrics import  accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn.model_selection import  train_test_split, cross_val_predict, StratifiedKFold
import time


data_test = "test.csv"
data_train="train.csv"

df= dc.data_to_pandas(data_train)
print(df.sample(5))
df_test=dc.data_to_pandas(data_test)
df['Title']= df['Name'].apply(lambda x: x.split(',')[-1].strip().split('.')[0])
df_test['Title']=df_test['Name'].apply(lambda x: x.split(',')[1].strip().split('.')[0])

title_dictionary = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir" : "Royalty",
    "Dr": "Officer",
    "Rev": "Officer",
    "the Countess":"Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr" : "Mr",
    "Mrs" : "Mrs",
    "Miss" : "Miss",
    "Master" : "Master",
    "Lady" : "Royalty"
}

df["Title"] = df["Title"].map(title_dictionary)
df_test["Title"] = df_test["Title"].map(title_dictionary)

x = df.drop(columns=["Survived", 'Name', 'Ticket', 'Cabin']) # X DATA (WILL BE TRAIN+VALID DATA)
y = df["Survived"] # 0 = No, 1 = Yes

x_test = df_test.drop(columns=['Name', 'Ticket', 'Cabin']) # # X_TEST DATA (NEW DATA)

cat_vars  = ['Sex', 'Embarked', 'Title']         # x.select_dtypes(include=[object]).columns.values.tolist()
num_vars  = ['Pclass', 'SibSp', 'Parch', 'Fare', 'Age'] # x.select_dtypes(exclude=[object]).columns.values.tolist()


def fit_pred_model():
    results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})

    x_train, x_val, y_train, y_val = train_test_split(x, y,random_state=0,test_size=0.2,stratify=y)

    for model_name, model in dc.model_types(num_vars, cat_vars):
        start_time = time.time()
    
        # FOR EVERY PIPELINE (PREPRO + MODEL) -> TRAIN WITH TRAIN DATA (x_train)
        model.fit(x_train, y_train)
        # GET PREDICTIONS USING x_val
        pred = model.predict(x_val)

        total_time = time.time() - start_time

        results = results.append({"Model":    model_name,
                              "Accuracy": accuracy_score(y_val, pred)*100,
                              "Bal Acc.": balanced_accuracy_score(y_val, pred)*100,
                              "Time":     total_time},
                              ignore_index=True)

    return results

    

print(fit_pred_model())

def cross_val_pred():
    results = pd.DataFrame({'Model': [], 'Accuracy': [], 'Bal Acc.': [], 'Time': []})

    """
    for model_name, model in tree_classifiers.items():
    start_time = time.time()
        
    # TRAIN AND GET PREDICTIONS USING cross_val_predict() and x,y
    pred = # CODE HERE

    total_time = time.time() - start_time

    results = results.append({"Model":    model_name,
                              "Accuracy": metrics.accuracy_score(y_val, pred)*100,
                              "Bal Acc.": metrics.balanced_accuracy_score(y_val, pred)*100,
                              "Time":     total_time},
                              ignore_index=True)
                              
                              
    """

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
    for model_name, model in dc.model_types(num_vars, cat_vars):

        start_time = time.time()
        pred = cross_val_predict(model, x, y, cv=skf)
        total_time = time.time() - start_time
        results = results.append({"Model":    model_name,
                              "Accuracy": accuracy_score(y, pred)*100,
                              "Bal Acc.": balanced_accuracy_score(y, pred)*100,
                              "Time":     total_time},
                              ignore_index=True)
    return results
