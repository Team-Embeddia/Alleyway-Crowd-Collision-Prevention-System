from flask import Flask, jsonify

app = Flask(__name__)

def run_http_server(shared_dict):
    @app.route('/person_count', methods=['GET'])
    def get_person_count():
        person_count = shared_dict.get("person_count", 0)
        return jsonify({"person_count": person_count})

    app.run(host="0.0.0.0", port=5000)

if __name__ == "__main__":
    from multiprocessing import Manager
    with Manager() as manager:
        shared_dict = manager.dict({"person_count": 0})
        run_http_server(shared_dict)
