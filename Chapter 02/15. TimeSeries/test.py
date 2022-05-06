import numpy as np
import pandas as pd
from sklearn.linear_model           import  LinearRegression
from sklearn.ensemble               import AdaBoostRegressor
from sklearn.ensemble               import GradientBoostingRegressor
from xgboost                        import XGBRegressor
from lightgbm                       import LGBMRegressor

import joblib
from get_features import GetFeature
from model_train import TrainModel

regressors = {
        "Linear"  : LinearRegression(),
        "AdaBoost":      AdaBoostRegressor(n_estimators=100),
        "Skl GBM":       GradientBoostingRegressor(n_estimators=100),
        "XGBoost":       XGBRegressor(n_estimators=100),
        "LightGBM":      LGBMRegressor(n_estimators=100),
        
    }

np.random.seed(0)
# Load the data
df=pd.read_csv(r'C:\Users\KINGSLEY\OneDrive\Documents\GitHub\Strive_school_Exercise\Chapter 02\15. TimeSeries\climate.csv')
#print(df)
# drop the data time column

df=df.drop(["Date Time"], axis=1)
print(df.shape)

gt=GetFeature(data=df, target_name='T (degC)')

target=gt.target # definding the target
clf=TrainModel(regressors)

# testing with diff regression model and predicting with mean sample features

x_mean= gt.get_feat_all_mean()

results_mean=clf.train_pred(x=x_mean, y=target)
print(f'{"*"*10} Training the model with x_mean feat: {"*"*10}\n {results_mean}')


# testing with diff regression model and predicting with standard devaition sample features

x_std=gt.get_feat_all_std()

result_std=clf.train_pred(x=x_std, y=target)
print(f'{"*"*10} Training the model with std feat: {"*"*10}\n {result_std}')

# testing with diff regression model and predicting with standard devaition and mean sample features

x_mean_std=gt.get_feat_std_mean()
res_mean_std=clf.train_pred(x=x_mean_std, y=target)
print(f'{"*"*10} Training the model with x_mean_std feat: {"*"*10}\n {result_std}')

# testing with diff regression model and predicting with diff statistics sample features

x_statistic=gt.feat_stat_each_col()
res_stat=clf.train_pred(x=x_statistic, y=target)
print(f'{"*"*10} Training the model with varius statistics feat: {"*"*10}\n {result_std}')





'''
********** Training the model with x_mean feat: **********
       Model       MSE       MAB   % error       Time
0    Linear  0.394014  0.440018  0.006631   0.094742
1  AdaBoost  0.800812  0.698692  0.013477  47.795187
2   Skl GBM  0.362104  0.422054  0.006094  81.491086
3   XGBoost  0.348288  0.410158  0.005861  10.489902
4  LightGBM  0.352487  0.410403  0.005932   0.592414
'''

'''
********** Training the model with std feat: **********
       Model        MSE       MAB   % error       Time
0    Linear  31.217966  4.120389  0.525378   0.065817
1  AdaBoost  10.343077  2.501253  0.174067  58.994240
2   Skl GBM   2.687522  1.164938  0.045229  79.675966
3   XGBoost   1.146716  0.816326  0.019298  11.714668
4  LightGBM   1.192915  0.803496  0.020076   0.584437

'''

'''
********** Training the model with x_mean_std feat: **********
       Model        MSE       MAB   % error       Time
0    Linear  31.217966  4.120389  0.525378   0.065817
1  AdaBoost  10.343077  2.501253  0.174067  58.994240
2   Skl GBM   2.687522  1.164938  0.045229  79.675966
3   XGBoost   1.146716  0.816326  0.019298  11.714668
4  LightGBM   1.192915  0.803496  0.020076   0.584437

'''

'''
********** Training the model with varius statistics feat: **********
       Model        MSE       MAB   % error       Time
0    Linear  31.217966  4.120389  0.525378   0.065817
1  AdaBoost  10.343077  2.501253  0.174067  58.994240
2   Skl GBM   2.687522  1.164938  0.045229  79.675966
3   XGBoost   1.146716  0.816326  0.019298  11.714668
4  LightGBM   1.192915  0.803496  0.020076   0.584437
'''


best_model=regressors.get('Linear')

joblib.dump(best_model, 'model.pkl')