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
    """
    A Class used for predicting, model fitting and  cross validation of score
    ....

    Attribtes:
    ---------
    x_train, x_test, y_train, y_test: -> array

    model: 
        A trained model with default= None

    """
    def __init__(self, x_train, x_test, y_train, y_test):
        """
        Parameters:
        ----------
        x_train, x_test, y_train, y_test: -> array

        """
        self.x_train, self.x_test=x_train, x_test
        self.y_train, self.y_test=y_train, y_test
        self.model=None

    def crossval(self):

        """
        Check and verifies the accuracy score of the train model
        print the mean cross validation of the model train
        """
        self.cv=cross_validate(self.model, self.x_train, self.y_train, return_estimator=True, cv=5)

        #print out cross validation scores
        [print('Crossvalidation fold: {}  Accruacy: {}'.format(n, score)) for n, score in enumerate(self.cv['test_score'])]
        #print out the mean of the cross validation
        print('Mean train cross validation score {}'.format(self.cv['test_score'].mean()))

    def predict(self, model):
        """
        Return the predict values of the test data

        parameter:

        Model:
            type of classification model (default is None)
        """
        self.predictions = model.predict(self.x_test)
        self.pred=(self.predictions == self.x_test).sum()/len(self.x_test)
        return self.pred

    def fit_model(self, model):
        """
        Return the fitted model

        parameter:
        ---------

        Model:
            type of classification model (default is None)

        """
        self.model=model.fit(self.x_train, self.y_train)
        return self.model

    def ac_score(self,model):
        """
        Return the  accuracy score of the fitted or trained model

        parameter:
        ---------

        Model:
            type of classification model (default is None)
        """
        self.ac=model.score(self.x_train,self.y_train)
        return self.ac


    def plot_matrix(self, model):
        """Return the  comfussion matrix of the train and fitted model

        parameter:
        ---------

        Model:
            type of classification model (default is None)
        """
        plot_confusion_matrix(model,self.x_test,self.y_test)



class ModelTrain(Main):
    """"""
    def __init__(self, x_train,y_train):
        super().__init__(x_train,x_test,y_train,y_test)
        self.model=None
    

    def randomforest(self):
        """"""
        self.model=RandomForestRegressor(n_estimators=100, max_depth=None, 
                                    min_samples_split=2, random_state=0)
       
        model=self.fit_model(self.model) # fitting the model
        accuracy=self.ac_score(self.model) #accuracy ofthe model
        print(f'Accuracy for random forest: {accuracy}')
        return model



    def adaboost(self):
        """"""
        self.model=AdaBoostRegressor(n_estimators=100, learning_rate=0.01, random_state=0)
        model=self.fit_model(self.model) # fitting the model
        accuracy=self.ac_score(self.model) #accuracy ofthe model
        print(f'Accuracy for Adaboost: {accuracy}')
        return model

    def gradient_boost(self):
        """"""
        model=GradientBoostingRegressor(learning_rate=0.01, n_estimators=100, random_state=0)
        model=self.fit_model(self.model) # fitting the model
        score=self.ac_score(self.model) #accuracy ofthe model
        print(f'Accuracy for Gradient Boost: {score}')
        return model


    def xboost(self):
        """"""
        model=XGBRegressor(n_estimators=100,learning_rating=0.001)
        model=self.fit_model(self.model) # fitting the model
        score=self.ac_score(self.model) #accuracy ofthe model
        print(f'Accuracy for Gradient Boost: {score}')
        return model



print(dh.hello)