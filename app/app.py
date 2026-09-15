from flask import Flask, render_template, request
import os

app = Flask(__name__,)
app.secret_key = os.urandom(16) 

@app.route('/')
def index():
    return render_template('index.html', test='')

@app.route('/ask', methods = ['POST'])
def search():
    if request.method == 'POST':
        if request.form['ask'] == 'Ask':
            question = request.form["question"].strip()
            

            return render_template('index.html', test=question)
    else:
        return render_template('index.html', test='')