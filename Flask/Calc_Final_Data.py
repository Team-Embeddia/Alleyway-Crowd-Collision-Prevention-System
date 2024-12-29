from flask import Flask, jsonify, send_file
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

def get_status():
    try:
        response = requests.get('http://100.118.65.82:5000/status')
        data = response.json()
        return data['people_count'], data['status']
    except requests.exceptions.RequestException as e:
        return None, None

def get_person_count():
    try:
        response = requests.get('http://localhost:5000/person_count')
        data = response.json()
        return data['person_count']
    except requests.exceptions.RequestException as e:
        return None

@app.route('/aggregate_status', methods=['GET'])
def aggregate_status():
    people_count_status, status = get_status()
    person_count = get_person_count()

    if people_count_status is not None and person_count is not None:
        average_people_count = round((people_count_status + person_count) / 2)
        return jsonify({
            "people_count": average_people_count,
            "status": status
        }), 200
    else:
        return jsonify({"error": "Unable to retrieve data"}), 500

@app.route('/get_image', methods=['GET'])
def get_image():
    response = send_file('/home/minjun/Desktop/image.jpg', mimetype='image/jpeg')
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
