from flask import Flask, jsonify, send_file, Response
import requests
import os
import json
from io import BytesIO

app = Flask(__name__)

REMOTE_SERVER_URL = "http://100.123.108.91:5001"

def load_results():
    try:
        with open("shared_data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"people_count": 0, "status": "Clear", "density": 0.0}

@app.route("/status", methods=["GET"])
def get_status():
    results = load_results()
    return jsonify(results)

@app.route('/get_image', methods=['GET'])
def get_image():
    try:
        response = requests.get(f"{REMOTE_SERVER_URL}/get_image", stream=True)
        if response.status_code == 200:
            return Response(
                response.content,
                content_type=response.headers.get('Content-Type', 'image/jpeg'),
            )
        return jsonify({"error": "Image not found on remote server"}), 404
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_image_output', methods=['GET'])
def get_image_output():
    image_path = os.path.expanduser(r"C:\Users\minjun\Desktop\Python\output.png")
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    return jsonify({"error": "Image not found"}), 404

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
