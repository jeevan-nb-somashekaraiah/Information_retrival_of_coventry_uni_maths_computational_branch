from flask import Flask, render_template, request
from search import search
from jinja2 import Environment
env = Environment(autoescape=True)
env.globals.update(len=len)


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    
    query = ''
    results = []
    documents = []
    publications = []
    if request.method == 'POST':
        query = request.form['query']
        results, documents,publications = search(query)
    return render_template('index.html', query=query, results=results, documents=documents,publications=publications,env=env)

if __name__ == '__main__':
    app.run(debug=True)

  