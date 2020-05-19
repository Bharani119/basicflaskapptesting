from flask import Flask, render_template
from datetime import datetime
import pandas as pd
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
        data_df = pd.DataFrame.from_dict(to_predict_list)

        # predictions
        result = model.predict(data_df)
        return render_template("result.html", prediction=result)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
