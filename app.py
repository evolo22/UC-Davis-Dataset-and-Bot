# app.py
from flask import Flask, request, jsonify, render_template
from Back_End import assistant
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os

app = Flask(__name__)

# Use Render's PostgreSQL database URL
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///chat_log.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

class ChatLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    user_message = db.Column(db.Text, nullable=False)
    predicted_intent = db.Column(db.String(100))
    confidence = db.Column(db.Float)
    bot_response = db.Column(db.Text)
    session_id = db.Column(db.String(100), index=True)

    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'user_message': self.user_message,
            'predicted_intent': self.predicted_intent,
            'confidence': self.confidence,
            'bot_response': self.bot_response,
            'session_id': self.session_id
        }

with app.app_context():
    db.create_all()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")
    session_id = data.get("session_id", "unknown")

    response, intent, confidence = assistant.process_message(user_message)

    # Log to database
    log_entry = ChatLog(
        user_message=user_message,
        predicted_intent=intent,
        confidence=confidence,
        bot_response=response,
        session_id=session_id
    )
    db.session.add(log_entry)
    db.session.commit()

    return jsonify({
        "reply": response,
        "intent": intent,
        "confidence": confidence 
    })

# Export endpoint for downloading logs
@app.route("/export_logs")
def export_logs():
    """Download all logs as CSV for retraining"""
    import csv
    from io import StringIO
    from flask import make_response
    
    logs = ChatLog.query.order_by(ChatLog.timestamp).all()
    
    si = StringIO()
    writer = csv.writer(si)
    writer.writerow(['timestamp', 'user_message', 'predicted_intent', 
                     'confidence', 'bot_response', 'session_id'])
    
    for log in logs:
        writer.writerow([
            log.timestamp.isoformat(),
            log.user_message,
            log.predicted_intent,
            log.confidence,
            log.bot_response,
            log.session_id
        ])
    
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=chat_logs.csv"
    output.headers["Content-type"] = "text/csv"
    return output

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)