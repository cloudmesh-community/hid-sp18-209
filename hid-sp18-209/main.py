#Author: Anthony Tugman
#Title:  E222 Final Project HID-SP18-209

#import required libraries and dependencies
import flask
from scipy import misc
import json
from flask import Flask, request, jsonify, Response
import pandas as pd
import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.externals import joblib

app = Flask(__name__)

###########################
#load data for 20% split (hard set to my test train set)
data20 = pd.read_csv('data/FDX.csv')
#prepare data for split
y20 = data20.Close
x20 = data20.drop(['Close', 'Date', 'Adj Close'], axis = 1)
y20 = y20.astype('int')

#actually splitting the data
x20_train, x20_test, y20_train, y20_test = train_test_split(x20, y20, test_size=0.2)

#performing regression
regressor = LinearRegression()
regressor.fit(x20_train, y20_train)
y_pred20 = regressor.predict(x20_test)
rscore20 = r2_score(y20_test, y_pred20)
print("R-Squared Score 20% test/80% train:")
print(rscore20)
print("")
###########################


###########################
#load data for 40% split (hard set to my test train set)
data40 = pd.read_csv('data/FDX.csv')

#prepare data for split
y40 = data40.Close
x40 = data40.drop(['Close', 'Date', 'Adj Close'], axis = 1)
y40 = y40.astype('int')

#actually splitting the data
x40_train, x40_test, y40_train, y40_test = train_test_split(x40, y40, test_size=0.4)

#performing regression
regressor = LinearRegression()
regressor.fit(x40_train, y40_train)
y_pred40 = regressor.predict(x40_test)
rscore40 = r2_score(y40_test, y_pred40)
print("R-Squared Score 40% test/60% train:")
print(rscore40)
print("")
###########################

##########################
#load data for 60% split (hard set to my test train set)
data60 = pd.read_csv('data/FDX.csv')

#prepare data for split
y60 = data60.Close
x60 = data60.drop(['Close', 'Date', 'Adj Close'], axis = 1)
y60 = y60.astype('int')

#actually splitting the data
x60_train, x60_test, y60_train, y60_test = train_test_split(x60, y60, test_size=0.6)

#performing regression
regressor = LinearRegression()
regressor.fit(x60_train, y60_train)
y_pred60 = regressor.predict(x60_test)
rscore60 = r2_score(y60_test, y_pred60)
print("R-Squared Score 60%test/40% train:")
print(rscore60)
print("")
##########################


##########################
#load data for 80% split (hard set to my test train set)
data80 = pd.read_csv('data/FDX.csv')

#prepare data for split
y80 = data80.Close
x80 = data80.drop(['Close', 'Date', 'Adj Close'], axis = 1)
y80 = y80.astype('int')

#actually splitting the data
x80_train, x80_test, y80_train, y80_test = train_test_split(x80, y80, test_size=0.8)

#performing regression
regressor = LinearRegression()
regressor.fit(x80_train, y80_train)
y_pred80 = regressor.predict(x80_test)
rscore80 = r2_score(y80_test, y_pred80)
print("R-Squared Score 80% test/20% train:")
print(rscore80)
print("")
##########################
#bar plot of rsquared scores
#plt.figure()
#x=['20% test data', '40% test data', '60% test data', '80% test data']
#values=[rscore20, rscore40, rscore60, rscore80]
#plt.bar(x, values, color = 'green')
#plt.xlabel("Percent training data")
#plt.ylabel("R2 score")
#plt.title("R2 score based on percent of training/test data")
#plt.savefig('graph/rscore.png')

##determine which ratio of train test is the most accurate
##display decision to user
##set trigger which decides which ratio to use for predictions
if rscore20 >= rscore40 and rscore20 >= rscore60 and rscore20 >= rscore80:
	print ('Using 20% test data is most accurate based on the r-squared value')
	greatest = rscore20
	trigger  = 1
	#rscore = y_pred20
elif rscore40 >= rscore20 and rscore40 >= rscore60 and rscore40 >= rscore80:
	print ('Using 40% test data is most accurate based on the r-squared value')
	greatest = rscore40
	trigger = 2
	#rscore = y_pred40
elif rscore60 >= rscore20 and rscore60 >= rscore40 and rscore60 >= rscore80:
    	print ('Using 60% test data is most accurate based on the r-squared value')
    	greatest = rscore60
    	trigger = 3
	#rscore = y_pred60
else:
    	print ('Using 80% test data is most accurate based on the r-squared value')
    	greatest = rscore80
    	trigger =4
	#rscore = y_pred80
   
##determine which ratio is the least accurate
##display decision to user
if rscore20 <= rscore40 and rscore20 <= rscore60 and rscore20 <= rscore80:
    print ('Using 20% test data is least accurate based on the r-squared value')
    least = rscore20
    
elif rscore40 <= rscore20 and rscore40 <= rscore60 and rscore40 <= rscore80:
    print ('Using 40% test data is least accurate based on the r-squared value')
    least = rscore40
elif rscore60 <= rscore20 and rscore60 <= rscore40 and rscore60 <= rscore80:
    print ('Using 60% test data is least accurate based on the r-squared value')
    least = rscore60
else:
    print ('Using 80% test data is least accurate based on the r-squared value')
    least = rscore80

##test user data using most accurate method
##this data will be predicted into the future based on 
##the number of data points entered
#now test user data
    
if trigger == 4:
	usertrain = pd.read_csv('data/FDX.csv')
	user50y = usertrain.Close
	user50x = usertrain.drop(['Close', 'Date', 'Adj Close'], axis = 1)
	user50y = user50y.astype('int')
	stick = regressor.fit(x80_train, y80_train)
	testpred = regressor.predict(user50x)
elif trigger == 3:
	usertrain = pd.read_csv('data/FDX.csv')
	user50y = usertrain.Close
	user50x = usertrain.drop(['Close', 'Date', 'Adj Close'], axis = 1)
	user50y = user50y.astype('int')
	stick = regressor.fit(x60_train, y60_train)
	testpred = regressor.predict(user50x)
elif trigger == 2:
	usertrain = pd.read_csv('data/FDX.csv')
	user50y = usertrain.Close
	user50x = usertrain.drop(['Close', 'Date', 'Adj Close'], axis = 1)
	user50y = user50y.astype('int')
	stick = regressor.fit(x40_train, y40_train)
	testpred = regressor.predict(user50x)
else:
	usertrain = pd.read_csv('data/FDX.csv')
	user50y = usertrain.Close
	user50x = usertrain.drop(['Close', 'Date', 'Adj Close'], axis = 1)
	user50y = user50y.astype('int')
	stick = regressor.fit(x20_train, y20_train)
	testpred = regressor.predict(user50x)

#save trained model
joblib.dump(stick, 'model.pkl')

#plot figures actual vs. predicted
#plt.figure()
#plt.plot(y20_test.values, label = "actual values")
#plt.plot(y_pred20, label = "predicted values")
#plt.ylabel("Closing Price ($)")
#plt.title("Display Prediction vs. Actual for 20% Training Data")
#plt.legend(loc= 'upper center')
#plt.savefig('graph/20test.png')

#plt.figure()
#plt.plot(y40_test.values, label = "actual values")
#plt.plot(y_pred40, label = "predicted values")
#plt.ylabel("Closing Price ($)")
#plt.title("Display Prediction vs. Actual for 40% Training Data")
#plt.legend(loc= 'upper center')
#plt.savefig('graph/40test.png')

#plt.figure()
#plt.plot(y60_test.values, label = "actual values")
#plt.plot(y_pred60, label = "predicted values")
#plt.ylabel("Closing Price ($)")
#plt.title("Display Prediction vs. Actual for 60% Training Data")
#plt.legend(loc= 'upper center')
#plt.savefig('graph/60test.png')

#plt.figure()
#plt.plot(y80_test.values, label = "actual values")
#plt.plot(y_pred80, label = "predicted values")
#plt.ylabel("Closing Price ($)")
#plt.title("Display Prediction vs. Actual for 80% Training Data")
#plt.legend(loc= 'upper center')
#plt.savefig('graph/80test.png')

#flask outputs
@app.route("/")
def home():
	return 'the model is trained visit /display for predictions, /r20 for rsquared20 score, /r40 for rsquare40 score, /r60 for rsquared60 score, and /r80 for rsquared80 score'
	
@app.route("/display")
def upload():
	###change test data here###
	testset = 'data/UPS.csv'
	df = pandas.read_csv(testset)
	df = df.drop(['Close', 'Date', 'Adj Close'], axis = 1)
	prediction = model.predict(df)
	prediction1 = model.predict(df)
	return jsonify({'prediction': list(prediction)})
	return(prediction)

@app.route("/r20")
def display20():
	r2 = rscore20
	return jsonify(r2)

@app.route("/r40")
def display40():
        r4 = rscore40
        return jsonify(r4)

@app.route("/r60")
def display60():
        r6 = rscore60
        return jsonify(r6)

@app.route("/r80")
def display80():
        r8 = rscore80
        return jsonify(r8)

if __name__ == "__main__":
	model = joblib.load('model.pkl')
	app.run(host = '0.0.0.0',  port =8080)
