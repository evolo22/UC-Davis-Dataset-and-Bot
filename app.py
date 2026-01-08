from flask import Flask, request, jsonify, render_template
from Back_End import assistant
import csv
import os

app = Flask(__name__)

CHAT_LOG_FILE = "chat_log.csv"

if not os.path.exists(CHAT_LOG_FILE):
    with open(CHAT_LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_message", "bot_response", "predicted_intent"])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    response, intent = assistant.process_message(user_message)

    with open(CHAT_LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([user_message, response, intent])

    return jsonify({"reply": response, "intent": intent})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)