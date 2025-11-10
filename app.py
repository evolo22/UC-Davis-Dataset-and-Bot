from flask import Flask, request, jsonify, render_template
from Back_End import assistant

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    response = assistant.process_message(user_message)
    return jsonify({"reply": response})

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000) 