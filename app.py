from flask import Flask, request, jsonify, render_template
from src.chatbot import answer

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    query = request.json.get("query", "")
    response = answer(query)
    return jsonify({"response": response})

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)