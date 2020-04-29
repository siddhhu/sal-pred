import flask
import pickle
import numpy as np
import pandas as pd
# Use pickle to load in the pre-trained model.
with open('employee.pkl', 'rb') as f:
    model = pickle.load(f)
app = flask.Flask(__name__, template_folder='templates')
@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('employee.html'))
    if flask.request.method == 'POST':
        experience = flask.request.form['experience']
        testscore = flask.request.form['testscore']
        skills = flask.request.form['skills']
        input_variables = pd.DataFrame([[experience, testscore, skills]],
                                       columns=['Experience', 'Test Score', 'Skills'],
                                       dtype=float)
        prediction = model.predict(input_variables)[0]
        return flask.render_template('employee.html',
                                     original_input={'Experience':experience,
                                                     'Test Score':testscore,
                                                     'Skills':skills},
                                     result=prediction,
                                     )
if __name__ == '__main__':
    app.run(debug=True)