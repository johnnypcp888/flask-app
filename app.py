import flask
import gzip, pickle
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Use pickle to load in the pre-trained model.
#with open('model/global.model','rb') as f:
    #model = pickle.load(f)
model = pd.read_pickle("model/model2.sav")
#f = gzip.open('model/model2.pklz','rb')
#f.close()

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    
    column_scaler = ['Debt_To_Income','Inquiries_Last_6Mo','Months_Since_Deliquency','Number_Open_Accounts_ratio_Total_Accounts', 'Loan_Amount_Requested_ratio_Annual_Income']
    scaler = StandardScaler()
    
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
    
        Inquiries_Last_6Mo = flask.request.form['Inquiries_Last_6Mo']
        Loan_Amount_Requested = flask.request.form['Loan_Amount_Requested']
        Annual_Income = flask.request.form['Annual_Income']
        
        Income_Verified_2 = flask.request.form['Income_Verified_2']
        Months_Since_Deliquency = flask.request.form['Months_Since_Deliquency']
        Debt_To_Income = flask.request.form['Debt_To_Income']
        
        Number_Open_Accounts = flask.request.form['Number_Open_Accounts']
        Total_Accounts = flask.request.form['Total_Accounts']
        Purpose_Of_Loan_1 = flask.request.form['Purpose_Of_Loan_1']
        
        Home_Owner_1 = flask.request.form['Home_Owner_1']
        Income_Verified_1 = flask.request.form['Income_Verified_1']
        Purpose_Of_Loan_3 = flask.request.form['Purpose_Of_Loan_3']
        
        Loan_Amount_Requested_ratio_Annual_Income = Loan_Amount_Requested/Annual_Income
        Number_Open_Accounts_ratio_Total_Accounts = Number_Open_Accounts/Total_Accounts
        
        input_variables = pd.DataFrame([[Inquiries_Last_6Mo, Loan_Amount_Requested_ratio_Annual_Income, Income_Verified_2, Months_Since_Deliquency, Debt_To_Income, Number_Open_Accounts_ratio_Total_Accounts, Purpose_Of_Loan_1, Home_Owner_1, Income_Verified_1, Purpose_Of_Loan_3]], columns=['Inquiries_Last_6Mo', 'Loan_Amount_Requested_ratio_Annual_Income', 'Income_Verified_2', 'Months_Since_Deliquency', 'Debt_To_Income', 'Number_Open_Accounts_ratio_Total_Accounts', 'Purpose_Of_Loan_1', 'Home_Owner_1', 'Income_Verified_1', 'Purpose_Of_Loan_3'],dtype=float)
        
        convert_dict = {'Income_Verified_2': int, 
                        'Purpose_Of_Loan_1': int,
                        'Home_Owner_1': int,
                        'Income_Verified_1': int,
                        'Purpose_Of_Loan_3': int
                       } 
        
        input_variables = input_variables.astype(convert_dict) 
        
        input_variables[column_scaler] = scaler.fit_transform(input_variables[column_scaler])
        d_train_X = xgb.DMatrix(input_variables)
        
        prediction = model.predict(d_train_X)[0]
        return flask.render_template('main.html',original_input={'Inquiries Last 6 months':Inquiries_Last_6Mo,'Loan Amount Requested':Loan_Amount_Requested,'Annual Income':Annual_Income, 'Income Not Verified':Income_Verified_2,'Months Since Delinquency':Months_Since_Deliquency,'Debt To Income':Debt_To_Income, 'Number of Open Accounts':Number_Open_Accounts,'Total number of Accounts':Total_Accounts,'Purpose of Loan - Credit Card':Purpose_Of_Loan_1, 'Home Owner - Rent':Home_Owner_1,'Income Source Verified':Income_Verified_1,'Other Purpose of Loan':Purpose_Of_Loan_3},result=prediction,)
    
if __name__ == '__main__':
    app.run()