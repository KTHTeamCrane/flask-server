from flask import Flask, request, jsonify
from inference import run_inference
import json

app = Flask(__name__)

@app.route("/echo")
def echo():
    if request.args.get("query") == None:
        return "To fake or not to fake, that is the question"
    
    return request.args["query"]

@app.route("/")
def help():
    help = """
method  endpoint    params

GET     echo        query
POST    api/fake    <body>
"""
    return help

@app.route("/api/fake", methods=["POST"])
def api_fake():
    if request.get_data() == "":
        return jsonify(error=True, error_msg="Article not provided", data=None)
    
    result = run_inference(str(request.get_data()))
    data = {
        "error": False,
        "error_msg": None,
        "data": result,
    }

    return json.dumps(data)
    

@app.route("/test")
def test_receive_body():
    return request.get_data()

if __name__ == "__main__":
    app.run(debug=True, port=6969, host="0.0.0.0")
