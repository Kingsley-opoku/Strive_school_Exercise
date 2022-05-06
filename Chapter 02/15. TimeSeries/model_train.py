
import pandas as pd
import time
import seaborn as sns
from sklearn.model_selection        import train_test_split
from sklearn.metrics                import mean_absolute_error, mean_squared_error
## Regression
from sklearn.linear_model           import  LinearRegression
from sklearn.ensemble               import AdaBoostRegressor
from sklearn.ensemble               import GradientBoostingRegressor
from xgboost                        import XGBRegressor
from lightgbm                       import LGBMRegressor




class TrainModel:
    def __init__(self, x, y) -> any:
        self.x=x
        self.y=y

    # examples of regression model
    def regression(self) -> dict:
        regressors = {
        "Linear"  : LinearRegression(),
        "AdaBoost":      AdaBoostRegressor(n_estimators=100),
        "Skl GBM":       GradientBoostingRegressor(n_estimators=100),
        "XGBoost":       XGBRegressor(n_estimators=100),
        "LightGBM":      LGBMRegressor(n_estimators=100),
        
        }
        return regressors.items()


    def train_pred(self) -> pd.DataFrame:
        x_train, x_val, y_train, y_val= train_test_split(self.x,self.y, random_state=0,test_size=0.2) 
        results = pd.DataFrame({'Model': [], 'MSE': [], 'MAB': [], " % error": [], 'Time': []})
        rang = abs(y_train.max()) + abs(y_train.min())
        for model_name, model in self.regression():
    
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