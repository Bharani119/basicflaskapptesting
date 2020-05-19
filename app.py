from flask import Flask, render_template, request
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
app = Flask(__name__)


# load model
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        to_predict = np.array(to_predict_list).reshape(1, 4)
        # data_df = pd.DataFrame.from_dict(to_predict_list)

        # predictions
        result = model.predict(to_predict)
        if int(result)== 1:
            prediction ='Congrats you will survive'
        else:
            prediction ='Sorry see you in Hell or Heaven'
        return render_template("result.html", prediction = prediction)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
