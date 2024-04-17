from flask import Flask, request, jsonify
from inference import run_inference
import json

app = Flask(__name__)

@app.route("/")
def echo():
    if request.args.get("query") == None:
        return "To fake or not fake, that is the question"
    else: return request.args["query"]


@app.route("/api/fake", methods=["POST"])
def api_fake():
    if request.args.get("article") == None:
        return jsonify(error=True, error_msg="Article not provided", data=None)
    else: 
        result = run_inference(request.args.get("article"))
        data = {
            "error": False,
            "error_msg": None,
            "data": {
                "false": int(result[0] * 100),
                "true": int(result[2] * 100)
            }
        }
        return json.dumps(data)


