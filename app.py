from flask import Flask, render_template, request
import pandas as pd
import pickle as pk

app = Flask(__name__)

# Load the model and scaler
model = pk.load(open('logreg_model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl','rb'))
data = pd.read_csv(r'C:\Users\parha\Documents\learning\ML_full_project\my\new_v\loan approval_flask\loan_approval_dataset.csv')

@app.route('/', methods=['GET', 'POST'])
def loan_prediction():
    prediction = ""
    if request.method == 'POST':
        # Get form data
        no_of_dep = int(request.form['no_of_dep'])
        grad = request.form['grad']
        self_emp = request.form['self_emp']
        Annual_Income = float(request.form['Annual_Income'])
        Loan_Amount = float(request.form['Loan_Amount'])
        Loan_Dur = int(request.form['Loan_Dur'])
        Cibil = int(request.form['Cibil'])
        Assets = float(request.form['Assets'])

        # Transform inputs to model format
        grad_s = 0 if grad == 'Graduated' else 1
        emp_s = 0 if self_emp == 'No' else 1

        pred_data = pd.DataFrame([[no_of_dep, grad_s, emp_s, Annual_Income, Loan_Amount, Loan_Dur, Cibil, Assets]],
                                 columns=['no_of_dependents', 'education', 'self_employed', 'income_annum', 'loan_amount', 'loan_term', 'cibil_score', 'Assets'])
        pred_data = scaler.transform(pred_data)
        predict = model.predict(pred_data)
        
        # Determine the prediction result
        if predict[0] == 1:
            prediction = 'Loan Is Approved'
        else:
            prediction = 'Loan Is Rejected'
    
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
