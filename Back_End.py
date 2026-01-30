import os
import json
import pandas as pd
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sentence_transformers import SentenceTransformer
import pickle
import gc

# Reduce PyTorch memory usage
torch.set_num_threads(1)

courses_df = pd.read_excel("electrical_and_computer_engineering.xlsx")

class ChatBotModel(nn.Module):
    def __init__(self, embedding_dim, output_size):
        super(ChatBotModel, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class ChatbotAssistant:
    def __init__(self, intents_path, function_mappings=None):
        self.model = None
        self.intents_path = intents_path
        self.intents = []
        self.intents_responses = {}
        self.function_mapping = function_mappings
        self.prev_flag = ""
        self.x = None
        self.y = None
        
        # Load sentence transformer with lighter model
        print("Loading sentence transformer...")
        self.encoder = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # Smaller model!
        self.encoder.to('cpu')
        self.embedding_dim = 384
        print("Sentence transformer loaded!")
        gc.collect() 
    
    def parse_intents(self):
        with open(self.intents_path, 'r') as f:
            intents_data = json.load(f)
        
        self.intents = []
        self.intents_responses = {}
        self.patterns = []
        self.pattern_intents = []
        
        for intent in intents_data['intents']:
            tag = intent['tag']
            if tag not in self.intents:
                self.intents.append(tag)
                self.intents_responses[tag] = intent['responses']
            
            for pattern in intent['patterns']:
                self.patterns.append(pattern)
                self.pattern_intents.append(tag)
    
    def prepare_data(self):
        print("Encoding training patterns with Sentence-BERT...")
        # Encode all patterns at once (much faster)
        embeddings = self.encoder.encode(
            self.patterns, 
            show_progress_bar=True,
            batch_size=32
        )
        labels = [self.intents.index(intent) for intent in self.pattern_intents]
        
        self.x = np.array(embeddings)
        self.y = np.array(labels)
        print(f"Prepared {len(self.x)} training examples with {self.embedding_dim}D embeddings")
    
    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.x, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatBotModel(self.embedding_dim, len(self.intents))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}: Loss: {running_loss / len(loader):.4f}")

    def save_model(self, model_path, data_path):
        torch.save(self.model.state_dict(), model_path)
        
        with open(data_path, 'wb') as f:
            pickle.dump({
                'intents': self.intents,
                'intents_responses': self.intents_responses,
                'embedding_dim': self.embedding_dim
            }, f)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
            self.intents = data['intents']
            self.intents_responses = data['intents_responses']
            self.embedding_dim = data['embedding_dim']

        self.model = ChatBotModel(self.embedding_dim, len(self.intents))
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.model.eval()

    def process_message(self, input_message):
        # Encode the user's message
        embedding = self.encoder.encode([input_message], show_progress_bar=False)[0]
        embedding_tensor = torch.tensor([embedding], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(embedding_tensor)
            probabilities = torch.softmax(predictions, dim=1)
            confidence, predicted_class_index = torch.max(probabilities, dim=1)
            confidence = confidence.item()
            predicted_class_index = predicted_class_index.item()

        predicted_intent = self.intents[predicted_class_index]
        print(f"Intent: {predicted_intent}, Confidence: {confidence:.2%}")
        
        self.prev_flag = predicted_intent

        # Handle responses
        if self.prev_flag in ["salutation"]:
            if predicted_intent == "no_response":
                response = "Glad I could help!"
            elif predicted_intent == "yes_response":
                response = "Sure, what can I help you with?"
            else:
                response = random.choice(self.intents_responses.get(predicted_intent, ["I'm not sure I understand that yet."]))
        elif self.function_mapping and predicted_intent in self.function_mapping:
            self.function_mapping[predicted_intent]()
            response = random.choice(self.intents_responses.get(predicted_intent, ["Okay!"]))
        elif predicted_intent in ["prerequisite_inquiry", "description_inquiry", "units_inquiry", "course_inquiry"]:
            response = handle_course_inquiry(predicted_intent, input_message)
        else:
            response = random.choice(self.intents_responses.get(predicted_intent, ["I'm not sure I understand that yet."]))

        return response, predicted_intent, confidence


def find_course_row(user_input):
    """Extracts course code and finds the corresponding row."""
    match = re.search(r'\b([A-Z]{2,4}\s?\d{2,3}[A-Z]?)\b', user_input.upper())
    if not match:
        return None, None
    course_code = match.group(1).replace(" ", "")
    for _, row in courses_df.iterrows():
        if course_code in row["Course"].replace(" ", ""):
            return course_code, row
    return course_code, None

def find_all_courses_per_title(user_input, courses_df):
    stop_words = {"what","do","i","need","for","can","take","without","have","any","prerequisites","classes","before","complete"}
    words = [w for w in re.findall(r'\w+', user_input.lower()) if w not in stop_words]

    if not words:
        return []

    query = words[-1]
    print("Query keyword:", query)

    matching_rows = courses_df[courses_df["Title"].str.lower().str.contains(query, na=False)]
    list_of_courses_for_title = matching_rows["Course"].tolist()
    
    return list_of_courses_for_title

def handle_course_inquiry(tag, user_input):
    course_code, row = find_course_row(user_input)
    list_of_courses_per_title = find_all_courses_per_title(user_input, courses_df)

    if row is None and len(list_of_courses_per_title) == 0:
        if course_code:
            first_digit_idx = next((i for i, ch in enumerate(course_code) if ch.isdigit()), None)
            if first_digit_idx is not None:
                if not course_code[first_digit_idx] == '0':
                    alt_course_code = course_code[:first_digit_idx] + '0' + course_code[first_digit_idx:]
                    alt_code, alt_row = find_course_row(alt_course_code)
                    if alt_row is not None:
                        course_code, row = alt_code, alt_row
                    else:
                        return f"Sorry, I couldn't find any information for {course_code}."
                else:
                    return f"Sorry, I couldn't find any information for {course_code}."
            else:
                return f"Sorry, I couldn't find any information for {user_input}."
        else:
            return "Please include a valid course code (like MAT 021A or ECS 036A)."

    if tag == "prerequisite_inquiry":
        if list_of_courses_per_title:
            results = []
            results.append("You will need to take:")
            for code in list_of_courses_per_title:
                found = False
                for _, row in courses_df.iterrows():
                    if code.replace(" ", "").upper() == row["Course"].replace(" ", "").upper():
                        results.append(f"{row['Course']} with these prerequisites: {row.get('Prerequisites', 'None listed')}\n")
                        found = True
                        break
                if not found:
                    results.append(f"Course {code} not found.")
            return "\n".join(results)
        else:
            return f"The prerequisites for {row['Course']} are: {row['Prerequisites']}"

    elif tag == "description_inquiry":
        return f"{row['Course']} — {row['Title']} {row['Units']}.\nDescription: {row['Course Description']}"

    elif tag == "units_inquiry":
        return f"{row['Course']} is worth {row['Units']}."
    
    elif tag == "course_inquiry":
        return f"What do you want to know about {course_code}?\nI can give you information on prerequisites, description, and units"

    else:
        return "I can help with prerequisites, descriptions, or units. Try asking again!"


# Training/Loading logic
if os.path.isfile("chatbot_model_sbert.pth") and os.path.isfile("chatbot_data_sbert.pkl"):
    print("Loading existing model...")
    assistant = ChatbotAssistant("intents.json")
    assistant.load_model("chatbot_model_sbert.pth", "chatbot_data_sbert.pkl")
    print("Model loaded successfully!")
else:
    print("Training new model...")
    assistant = ChatbotAssistant("intents.json")
    assistant.parse_intents()
    assistant.prepare_data()
    assistant.train_model(batch_size=8, lr=0.001, epochs=50)
    assistant.save_model("chatbot_model_sbert.pth", "chatbot_data_sbert.pkl")
    print("Model trained and saved!")

print("Chatbot backend ready for Flask server.")
