#from pyexpat import model
from pyexpat import model

from scipy import rand
import data_handler as dh

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import joblib




#
x_train, x_test, y_train, y_test, ct = dh.get_data("./insurance.csv")


class Main:
    def __init__(self, x_train, x_test, y_train, y_test ):
        self.x_train, self.x_test=x_train, x_test
        self.y_train, self.y_test=y_train, y_test
        self.model=None

    def crossvalidate(self):
        self.cv=cross_validate(self.model, self.x_train, self.y_train, return_estimator=True, cv=5)

        #print out cross validation scores
        [print('Crossvalidation fold: {}  Accruacy: {}'.format(n, score)) for n, score in enumerate(self.cv['test_score'])]
        #print out the mean of the cross validation
        print('Mean train cross validation score {}'.format(self.cv['test_score'].mean()))

    def predict(self):
        predictions = self.model.predict(self.x_test)
        pred=(predictions == self.x_test).sum()/len(self.x_test)
        return pred
        

    def plot_matrix(self, model):
        plot_confusion_matrix(self.model,self.x_test,self.y_test)

  
class Mo(Main):
    def __init__(self, x_train, x_test, y_train, y_test):
        super().__init__(x_train, x_test, y_train, y_test)

def randomforest():
    model=RandomForestRegressor(n_estimators=100, max_depth=None, 
                                    min_samples_split=2, random_state=0)
    model=model.fit(x_train, y_train) # fit the data
    # accuracy=model.score(x_train, y_train)
    # print(f'Accuracy for random forest: {accuracy}')
    return model
model=randomforest()


def adaboost():
    model=AdaBoostRegressor(n_estimators=100, learning_rate=0.01, random_state=0)
    model.fit(x_train, y_train)
    accuracy=model.score(x_train, y_train)
    print(f'Accuracy for Adaboost: {accuracy}')
    return model



def gradient_boost():
    model=GradientBoostingRegressor(learning_rate=0.01, n_estimators=100, random_state=0)
    model.fit(x_train, y_train)
    score=model.score(x_train, y_train)
    print(f'Accuracy for Gradient Boost: {score}')
    return model




print(dh.hello)