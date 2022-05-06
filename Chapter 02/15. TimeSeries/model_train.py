
import pandas as pd
import time
import seaborn as sns
from sklearn.model_selection        import train_test_split
from sklearn.metrics                import mean_absolute_error, mean_squared_error
                    



class TrainModel:
    def __init__(self, regressor) -> any:
       self.regressor=regressor

    def train_pred(self, x, y) -> pd.DataFrame:
        x_train, x_val, y_train, y_val= train_test_split(x, y, random_state=0,test_size=0.2) 
        results = pd.DataFrame({'Model': [], 'MSE': [], 'MAB': [], " % error": [], 'Time': []})
        rang = abs(y_train.max()) + abs(y_train.min())
        for model_name, model in self.regressor.items():
    
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