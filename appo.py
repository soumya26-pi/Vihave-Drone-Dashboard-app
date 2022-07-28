from flask import Flask,request
from flask import Flask, request, jsonify
import numpy as np


app=Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World'

@app.route('/frame', methods=['POST'])
def predict():
    output=request.form.get("frame")
    return jsonify({"frame":str(output)})





if __name__ == '__main__':
    app.run(debug=True)