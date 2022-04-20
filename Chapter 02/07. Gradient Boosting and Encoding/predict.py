from test import optimal_model 

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer


while True:
    inputs=[]
    age = int(input("How old are you? \n"))
    child = int(input("How many children do you have? \n"))
    smoke = bool(input("Do you smoke? \n"))
    bmi=float(input('enter your BMI? \n'))
    sex=str(input('What is your sex? \n'))
    region=str(input('What region are you from? option: [northwest, southwest, northeast, southeast]\n'))
    inputs.append([age, sex, bmi, child, smoke, region])
    '''
    Preprocess
    predict
    
    '''
    ct = ColumnTransformer([('ordinal', OrdinalEncoder(handle_unknown= 'use_encoded_value', unknown_value = -1), [1,4,5] ), 
                                        ('numerical', StandardScaler(), [0, 2])], remainder='passthrough')

    pred=ct.fit_transform(inputs)
    model=optimal_model()
    print(model.predict(pred))



    
