from flask import Flask, jsonify, request
import time

app = Flask(__name__)


@app.route('/')
def index():
    return 'OK!'


@app.route("/fruit_detection", methods=['POST'])
def response():
    query = dict(request.form)['query']
    result = query + " " + time.ctime()
    return jsonify({"response": result})


if __name__ == "__main__":
    app.run()
