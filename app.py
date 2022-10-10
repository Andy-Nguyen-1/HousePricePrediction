from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import json
import plotly
import plotly.express as px
# Load pickle model
model = pickle.load(open('rfr_model_1.pkl', 'rb'))

# Create flask app
app = Flask(__name__)


@app.route("/home")

@app.route("/")
def home():
		# Load Homepage
		return render_template("base.html")

@app.route('/predict', methods=['POST'])
# Prediction Function
def predict():
	# Taking user input into the model
	bedrooms = request.form['bedrooms_grouped']
	bathrooms = request.form['bathrooms']
	condition = request.form['condition']
	sqft_living = request.form['sqft_living']
	arr = np.array([[bedrooms, bathrooms, condition, sqft_living]])
	pred = model.predict(arr)
	# Return to main page and show prediction
	return render_template('home.html', pred_text = "Your home is estimated at ${} ".format(pred))
@app.route("/visualize")
# Visualization function
def visualize():
	# Graph one
	df = pd.read_csv('new_data1.csv')
	fig = px.bar(df, x = 'bedrooms_grouped', y=['price'], title="Average price for house with number of bedrooms")
	graphJSON= json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

	#graph 2
	df = pd.read_csv('new_data1.csv')
	fig2 = px.histogram(df, x='price', title="Homes Price Distribution")
	graph2JSON = json.dumps(fig2, cls=plotly.utils.PlotlyJSONEncoder)

	#graph 3
	df = pd.read_csv('new_data1.csv')
	fig3 = px.scatter(df, x='sqft_living', y='price', title="Price vs Sq Ft Living", )
	graph3JSON = json.dumps(fig3, cls=plotly.utils.PlotlyJSONEncoder)
	return render_template('visualize.html', graphJSON=graphJSON, graph2JSON=graph2JSON, graph3JSON=graph3JSON)

# Run the Flask app
if __name__ == "__main__":
	app.run(debug=True) 