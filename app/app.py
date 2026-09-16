from flask import Flask, render_template, request
import os

from model.rag import RentalRAG

model = RentalRAG()

app = Flask(__name__,)
app.secret_key = os.urandom(16) 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods = ['POST'])
def search():
    if request.method == 'POST':
        if request.form['ask'] == 'Ask':
            question = request.form["question"].strip()

            answer = model.ask(question,top_k=5)

            return render_template('index.html', question = question, answer=answer)
    else:
        return render_template('index.html')