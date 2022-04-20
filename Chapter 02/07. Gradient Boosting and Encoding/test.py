import train 
from train import Main, ModelTrain
import data_handler as dh
import joblib
import pickle

x_train, x_test, y_train, y_test, ct= dh.get_data("./insurance.csv")
model=ModelTrain(x_train, y_train)
forest=model.randomforest()



def optimal_model():
    best_score=0
    model=ModelTrain(x_train, y_train)
    forest=model.randomforest()
    ad_boost=model.adaboost()
    grad_boost=model.gradient_boost()
    xg_boost=model.xboost()
    model_list=[forest, ad_boost,grad_boost,xg_boost]
    for best in model_list:
        if best.score(x_train,y_train)>best_score:
            best_score=best
        return f'The Optimal Model is: {best_score}'


print(optimal_model())

# file_name='model.sav'
# pickle.dump(forest, open(file_name, 'wb'))
# file_name='model.joblib'
# joblib.dump(forest, file_name)
# train.adaboost()
# train.gradient_boost()

# ma=main(x_train, x_test, y_train, y_test)
# print(ma.x_train.shape)
