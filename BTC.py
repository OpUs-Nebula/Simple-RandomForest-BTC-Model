import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import Imputer
from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier

main_file_path = "C:\\Users\\Mbwenga\\Downloads\\btc-usd-max.csv"

data = pd.read_csv(main_file_path,index_col=0)

##data_predictor/Prediction
DataPoints = ["market_cap","total_volume"]

X = data[DataPoints].drop(data[DataPoints].index[0])


#Shift for future prediction based on current data.
Price = data.price.shift(1) 
Y = Price.drop(Price.index[0])

train_X, val_X, train_y, val_y = train_test_split(X, Y,random_state = 0)

my_imputer = Imputer()

imputed_X_train = my_imputer.fit_transform(train_X)
imputed_X_test = my_imputer.transform(val_X)

def Random_BTC_Predict (X_Trn, Y_Trn, X_Val, Y_Val):
    model = RandomForestRegressor() 
    model.fit(X_Trn, Y_Trn)   
    Prds = model.predict(X_Val)
    mae = mean_absolute_error(Y_Val, Prds)

    #Returning error value
    return(mae)

print((Random_BTC_Predict(imputed_X_train, train_y, imputed_X_test, val_y)))


#Why this is useless? market_Cap = price * total_volume, ==> price = market_cap / total_volume. All information is already contained in DataPoints.

##Note, total volume is on a daily basis. not amount of bitcoins in circulation.I.E Above conclusion is not 100% correct##

#Why this isnt useless? The same principal goes for everything, Only that more advanced problems have more advanced equations describing them.

#Potential Fix? use input of code calculating expected future market_cap and total_volume from second algorithm, and its all good!
#(As if its an easy problem, lol)

#Exec: python C:\Users\Mbwenga\Desktop\BTC.py

#print(data.dtypes)