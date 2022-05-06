import numpy as np
import pandas as pd
from get_features import GetFeature
from model_train import TrainModel

# Load the data
df=pd.read_csv(r'C:\Users\KINGSLEY\OneDrive\Documents\GitHub\Strive_school_Exercise\Chapter 02\15. TimeSeries\climate.csv')
#print(df)
# drop the data time column

df=df.drop(["Date Time"], axis=1)
print(df.shape)

gt=GetFeature(data=df, target_name='T (degC)')

y= gt.get_target()

# testing with diff regression model and predicting with mean sample features

x_mean= gt.get_feat_all_mean()

models_mean=TrainModel(x=x_mean,y=y)

results_mean=models_mean.train_pred()
print(results_mean)

# testing with diff regression model and predicting with standard devaition sample features

x_std=gt.get_feat_all_std()

mod_std=TrainModel(x=x_std, y=y)

result_std=mod_std.train_pred()
print(result_std)
