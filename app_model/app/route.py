import flask
from flask import Flask

app = Flask(__name__)

@app.route('/models/i2v.json', methods=['POST'])
def model_i2v():
    return 'post'

@app.route('/healthcheck', methods=['GET', 'POST'])
def healthcheck():
    return ''

if __name__ == "__main__":
    app.run()

