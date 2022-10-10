from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import sklearn


# Load pickle model
model = pickle.load(open('rfr_model_1.pkl', 'rb'))



# Create flask app
app = Flask(__name__)


@app.route("/home")

@app.route("/")
def home():

		return render_template("base.html")


@app.route('/predict', methods=['POST'])
def predict():

	bedrooms = request.form['bedrooms_grouped']
	bathrooms = request.form['bathrooms']
	condition = request.form['condition']
	sqft_living = request.form['sqft_living']
	arr = np.array([[bedrooms, bathrooms, condition, sqft_living]])
	pred = model.predict(arr)
	
	return render_template('home.html', pred_text = "Your home is estimated at ${} ".format(pred))




if __name__ == "__main__":
	app.run(debug=True) 