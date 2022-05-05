# importting libraries
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import time
import seaborn as sns
from sklearn.model_selection        import train_test_split
from sklearn.metrics                import mean_absolute_error, mean_squared_error
## Regression
from sklearn.linear_model           import  LinearRegression
from sklearn.tree                   import DecisionTreeRegressor
from sklearn.ensemble               import RandomForestRegressor
from sklearn.ensemble               import ExtraTreesRegressor
from sklearn.ensemble               import AdaBoostRegressor
from sklearn.ensemble               import GradientBoostingRegressor
from xgboost                        import XGBRegressor
from lightgbm                       import LGBMRegressor
from catboost                       import CatBoostRegressor
# load data with pandas 
df=pd.read_csv(r'C:\Users\KINGSLEY\OneDrive\Documents\GitHub\Strive_school_Exercise\Chapter 02\15. TimeSeries\climate.csv')
#print(df)
# drop the data time column
df=df.drop(["Date Time"], axis=1)
print(df.shape)

def paring(data, seq_len=6):
    x=[]
    y=[]
    for i in range(0, (data.shape[0]-(seq_len+1)), seq_len+1):
        seq= np.zeros((seq_len, data.shape[1]))
        for j in range(seq_len):
            seq[j]=data.values[i+j]

        x.append(seq.flatten())
        y.append(data["T (degC)"][i+seq_len])
    
    return np.array(x), np.array(y)
# unparking x, y 
x, y=paring(df)
print(y)
print(x.shape)

# examples of regression model
def regression() -> dict:
    regressors = {
       "Linear"  : LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Extra Trees":   ExtraTreesRegressor(n_estimators=100),
    "Random Forest": RandomForestRegressor(n_estimators=100),
    "AdaBoost":      AdaBoostRegressor(n_estimators=100),
    "Skl GBM":       GradientBoostingRegressor(n_estimators=100),
    "XGBoost":       XGBRegressor(n_estimators=100),
    "LightGBM":      LGBMRegressor(n_estimators=100),
    "CatBoost":      CatBoostRegressor(n_estimators=100,allow_writing_files=False),
}
    return regressors.items()


def train_pred(x, y)-> pd.DataFrame:
    x_train, x_val, y_train, y_val= train_test_split(x,y, random_state=0,test_size=0.2) 
    results = pd.DataFrame({'Model': [], 'MSE': [], 'MAB': [], " % error": [], 'Time': []})
    rang = abs(y_train.max()) + abs(y_train.min())
    for model_name, model in regression():
    
        start_time = time.time()
        model.fit(x_train, y_train)
        total_time = time.time() - start_time
        
        pred = model.predict(x_val)
    
        results = results.append({"Model":    model_name,
                              "MSE": mean_squared_error(y_val, pred),
                              "MAB": mean_absolute_error(y_val, pred),
                              " % error": mean_squared_error(y_val, pred) / rang,
                              "Time":     total_time},
                              ignore_index=True)

    return results

#print(train_pred(x,y))


# selected model (Linear)
x_train, x_test, y_train, y_test= train_test_split(x,y, random_state=0,test_size=0.2)
linear=LinearRegression()
linear.fit(x_train, y_train)

print(linear.predict(x_test))
