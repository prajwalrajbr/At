from flask import Flask, render_template, request
import csv
import itertools
#iterables, which are objects that can be looped over, such as lists, tuples, and strings
# generating combinations of hyperparameters to be used in model training.
import random
#spilt data
import sqlite3
from ibm_hr_attrition import Traning_1
##from ibm_hr_attrition_ann import Traning_2
from test_attrition import Testing_1
import pandas as pd

app = Flask(__name__)

##template_folder='my_templates')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/graph')
def graph():
    return render_template('graph.html')

#@app.route is a decorator that is used to define the URL routes that our application will respond to
#special function that can be used to modify the behavior of another function or class
@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()
        #database object used to execute queries and fetch data from the database
        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()
        #retrieve all the rows returned by the query as a list of tuples

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('log.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()
        # commit changes made to a database after performing a transaction.

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')

@app.route('/traindata', methods=['GET', 'POST'])
def traindata():
    if request.method == 'POST':
        file1 = request.form['file1']
        #get the path of the CSV file
        print(file1)
        result = []
        f = open(file1, 'r')
        Data = csv.reader(f)
        for row in Data:
            result.append(row) 
        f.close()   
        return render_template('log.html',  train_result=result, file1=file1)

    return render_template('index.html')

@app.route('/testdata', methods=['GET', 'POST'])
def testdata():
    if request.method == 'POST':
        file1 = request.form['file1']
        print(file1)

        result = []
        f = open(file1, 'r')
        Data = csv.reader(f)
        for row in Data:
            result.append(row) 
        f.close()   
        return render_template('log.html',  test_result=result, file1=file1)

    return render_template('index.html')

@app.route('/training/<File>')
def training(File):
    print(File)
    model_names, acc = Traning_1(File)
    print(model_names, acc)
##    Traning_2(File)
    f = open('model_select.txt', 'w')
    if acc[0] > acc[1] and acc[0] > acc[2]:
        f.write('boost.pkl')
    elif acc[1] > acc[0] and acc[1] > acc[2]:
        f.write('rf.pkl')
    else:
        f.write('svm.pkl')
    f.close()
    return render_template('log.html', info='Training completed', model_names=model_names, acc=acc)

@app.route('/testing/<File>')
def testing(File):
    print(File)
    Testing_1(File)
    # Traning_2(File)

    result = []
    f = open('result.csv', 'r')
    Data = csv.reader(f)
    for row in Data:
        result.append(row) 
    f.close() 
    return render_template('log.html', info1='Testing completed', test_output=result)

@app.route('/Prediction')
def Prediction():
    return render_template('log.html')



@app.route('/generate_graph', methods=['GET', 'POST'])
def generate_graph():
    if request.method == 'POST':

        tcn = request.form['tcn']

        import warnings
        warnings.filterwarnings('always')
        warnings.filterwarnings('ignore')
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib import style
        import missingno as msno
        import seaborn as sns

        r='WA_Fn-UseC_-HR-Employee-Attrition.csv'
        df=pd.read_csv(r)

        plt.figure(figsize=(20,10))
        sns.countplot(x=df[tcn],data=df,hue='Attrition',palette="twilight_shifted",saturation=2,dodge=True,)

        plt.savefig('static/'+tcn+'.png')

        image = 'http://127.0.0.1:5000/static/'+tcn+'.png'

        return render_template('graph.html', image=image, tcn=tcn)

    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True,use_reloader=False)
# change is made