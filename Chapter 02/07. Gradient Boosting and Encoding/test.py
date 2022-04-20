import train 
from train import main
import data_handler as dh
import joblib
import pickle

x_train, x_test, y_train, y_test = dh.get_data("./insurance.csv")
forest=train.randomforest()
# file_name='model.sav'
# pickle.dump(forest, open(file_name, 'wb'))
# file_name='model.joblib'
# joblib.dump(forest, file_name)
# train.adaboost()
# train.gradient_boost()

# ma=main(x_train, x_test, y_train, y_test)
# print(ma.x_train.shape)