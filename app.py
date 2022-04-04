from flask import Flask, render_template, url_for, request
import pickle
import numpy as np
# import pandas as pd

app = Flask(__name__)
model = pickle.load(open('models/model.pkl', 'rb'))


@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/quiz')
def quiz():
    return render_template('Quiz.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():

    int_features = [eval(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    input_data_reshaped = final_features
    # column_names = ['age', 'gender', 'E1', 'E2', 'E3',
    #                 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'N1', 'N2', 'N3', 'N4', 'N5',
    #                 'N6', 'N7', 'N8', 'N9', 'N10', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7',
    #                 'A8', 'A9', 'A10', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9',
    #                 'C10', 'O1', 'O2', 'O3', 'O4', 'O5', 'O6', 'O7', 'O8', 'O9', 'O10'
    #                 ]
#     df = pd.DataFrame(columns=column_names)
#     d = {}
#     def insert_me():
#         for i in range(len(column_names)):
#             d[column_names[i]] = final_features[0][i]
#         return d
#
#     df = df.append(insert_me(), ignore_index=True)

    prediction = model.predict(input_data_reshaped)

    return render_template('after.html', data= prediction)

if __name__ == '__main__':
    app.run(debug=True)
