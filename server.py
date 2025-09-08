from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # strokes from Roblox
    # For now, we ignore strokes and always say "cat"
    return jsonify({"guess": "cat"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
