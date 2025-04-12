from flask import Flask, render_template, request, send_from_directory, jsonify, redirect, url_for, session
import os, random
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from pymongo import MongoClient
from openai import OpenAI

from functions import (
	img_predict,
	get_diseases_classes,
	get_crop_recommendation,
	get_fertilizer_recommendation,
	soil_types,
	Crop_types,
	crop_list
)

# Load environment variables from .env
load_dotenv()

# Base directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Flask app initialization
app = Flask(__name__)
random.seed(0)

# Secret key for sessions
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'mydefaultdevkey')

# Upload folders
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# OpenAI Client Setup
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-your-key"))

# MongoDB Connection
mongo_client = MongoClient(os.environ.get("MONGO_URI"))
db = mongo_client["urban_agri"]
users_collection = db["users"]

@app.route('/')
def index():
	return render_template('index.html')

# ------------------ Login ---------------------
@app.route('/login', methods=['GET', 'POST'])
def login():
	if request.method == 'POST':
		username = request.form['username']
		password = request.form['password']
		user = users_collection.find_one({"username": username})
		if user and user['password'] == password:
			session['username'] = username
			return redirect(url_for('index'))
		else:
			return render_template('login.html', error="Invalid username or password")
	return render_template('login.html')

@app.route('/logout')
def logout():
	session.pop('username', None)
	return redirect(url_for('index'))

# ------------------ Register ---------------------
@app.route('/register', methods=['GET', 'POST'])
def register():
	if request.method == 'POST':
		username = request.form['username']
		password = request.form['password']

		if users_collection.find_one({"username": username}):
			return render_template('register.html', error="Username already exists.")

		users_collection.insert_one({"username": username, "password": password})
		session['username'] = username
		return redirect(url_for('index'))
	return render_template('register.html')
# --------------------------------------------------

@app.route('/crop-recommendation', methods=['GET', 'POST'])
def crop_recommendation():
	if request.method == "POST":
		try:
			to_predict_list = request.form.to_dict()
			to_predict_list = list(map(float, to_predict_list.values()))
			result = get_crop_recommendation(to_predict_list)
			return render_template("recommend_result.html", result=result)
		except Exception as e:
			return f"Error in crop recommendation: {e}"
	else:
		return render_template('crop-recommend.html')

@app.route('/fertilizer-recommendation', methods=['GET', 'POST'])
def fertilizer_recommendation():
	if request.method == "POST":
		try:
			to_predict_list = request.form.to_dict()
			to_predict_list = list(map(float, to_predict_list.values()))
			result = get_fertilizer_recommendation(
				num_features=to_predict_list[:-2],
				cat_features=to_predict_list[-2:]
			)
			return render_template("recommend_result.html", result=result)
		except Exception as e:
			return f"Error in fertilizer recommendation: {e}"
	else:
		return render_template(
			'fertilizer-recommend.html',
			soil_types=enumerate(soil_types),
			crop_types=enumerate(Crop_types)
		)

@app.route('/crop-disease', methods=['POST', 'GET'])
def find_crop_disease():
	if request.method == "GET":
		return render_template('crop-disease.html', crops=crop_list)
	else:
		try:
			file = request.files["file"]
			crop = request.form["crop"]
			filename = secure_filename(file.filename)
			file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			file.save(file_path)

			prediction = img_predict(file_path, crop)
			result = get_diseases_classes(crop, prediction)

			return render_template('disease-prediction-result.html', image_file_name=filename, result=result)
		except Exception as e:
			return f"Error in crop disease prediction: {e}"

@app.route('/uploads/<filename>')
def send_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ----------------- AI Chat Assistant Endpoint -------------------
@app.route('/chat', methods=['POST'])
def chat():
	try:
		user_message = request.json.get("message")
		if not user_message:
			return jsonify({"response": "Please enter a message."})

		response = client.chat.completions.create(
			model="gpt-3.5-turbo",
			messages=[
				{"role": "system", "content": "You are an expert AI assistant for agriculture. Help users with soil, crop, fertilizer, weather, and plant disease information."},
				{"role": "user", "content": user_message}
			]
		)

		reply = response.choices[0].message.content
		return jsonify({"response": reply})

	except Exception as e:
		return jsonify({"response": f"Something went wrong: {str(e)}"})
# ----------------------------------------------------------------

if __name__ == '__main__':
	app.run(debug=True)
