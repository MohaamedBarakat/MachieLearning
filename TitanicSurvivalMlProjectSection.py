# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 17:23:40 2019

@author: MBarakat
"""

import pandas as pd
import numpy as np
#read data form excel file
data = pd.read_csv('titanic_data.csv')
#drop some un-wanted data
dataFilter = data.drop(['PassengerId','Name','Ticket','Cabin'],axis = 1)
#dataFilter = dataFilter.where((pd.notnull(dataFilter)), None)

#split features (independent variable)
X = dataFilter.iloc[:,1:8].values
#split target data ( dependent variable) 
y = dataFilter.iloc[:,0].values
#Take care of missing value
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(X[:,2:6])
X[:,2:6] = imputer.transform(X[:,2:6])
#encode the gender column to discrete data
from sklearn.preprocessing import  LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,1] = labelencoder_X.fit_transform(X[:,1])
X[:,6] = labelencoder_X.fit_transform(X[:,6])
onehotencoder = OneHotEncoder(categorical_features = [1,6])
X = onehotencoder.fit_transform(X).toarray()
#d = pd.DataFrame(X)

#calculate number of classe no
no = 0
for i in y:
    if i == 0:
        no+=1
#number of yes
yes = len(y) - no
#print(c)

#splitting data to test
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0, shuffle=False ,stratify=None)

#fetures rescaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#from sklearn.neural_network import MLPClassifier
#nn = MLPClassifier(hidden_layer_sizes = (100,5),activation="relu",solver='adam',alpha=0.0001,random_state=0,shuffle=False)
#nn.fit(X_train, y_train)
#y_pred = nn.predict(X_test)
#cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score

def NaiveBayes(X_train, X_test, y_train, y_test):
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    return accuracy_score(y_test, y_pred)
def SVM(X_train, X_test, y_train, y_test):
    from sklearn.svm import SVC
    clf1 = SVC(kernel='linear')
    clf1.fit(X_train, y_train)
    y_pred = clf1.predict(X_test)
    return accuracy_score(y_test, y_pred)
def DecisionTree(X_train, X_test, y_train, y_test):
    from sklearn import tree
    dt = tree.DecisionTreeClassifier(random_state=0)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    return accuracy_score(y_test, y_pred)
def NeuralNetwork(X_train, X_test, y_train, y_test):
    from sklearn.neural_network import MLPClassifier
    nn = MLPClassifier(hidden_layer_sizes = (100,5),activation="relu",solver='adam',alpha=0.0001,random_state=0,shuffle=False)
    nn.fit(X_train, y_train)
    y_pred = nn.predict(X_test)
    return accuracy_score(y_test, y_pred)
# back button in html file
backButton =" % </h1> <br> <button onclick='goBack()'> Go Back </button></center><script>function goBack() {window.history.back();}</script>"
#data to visual in html
dataVisu = np.array(data)
dataVisu = list(dataVisu[:10,:])
from flask import Flask,render_template,request
app = Flask(__name__)
@app.route('/')
def main():
    return render_template('ML.html')
@app.route('/result',methods=['POST','GET'])
def index():
    value = request.form['select1']
    if value =="NB" :
        return "<center> <h1> accuracy score of Naive Bayes = " +  str(NaiveBayes(X_train, X_test, y_train, y_test)*100) + backButton
    elif value == "NN":
        return " <center> <h1> accuracy score of Neural Network = " + str(NeuralNetwork(X_train, X_test, y_train, y_test)*100) + backButton
    elif value == "DT" :
        return "<center> <h1> accuracy score of Decision Tree = " + str(DecisionTree(X_train, X_test, y_train, y_test)*100) + backButton
    else :
        return "<center> <h1> accuracy score of Support Vector Machine = " + str(SVM(X_train, X_test, y_train, y_test)*100) + backButton
@app.route('/table')
def table():
    return  str(data.iloc[:20,:]) + "<br> <button onclick='goBack()'> Go Back </button></center><script>function goBack() {window.history.back();}</script>"

if __name__ == "__main__":
    app.run(debug=True)
#print(NeuralNetwork(X_train, X_test, y_train, y_test))












