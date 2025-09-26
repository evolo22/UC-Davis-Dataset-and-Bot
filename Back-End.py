import os
import json
import random
import nltk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#Need to run (uncomment below) this the first time!
# nltk.download('punkt_tab')
# nltk.download('wordnet')

class ChatBotModel(nn.Module):

    def __intit__(self, input_size, output_size):
        super(ChatBotModel, self).__init__()

        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)

        x = self.relu(self.fc3(x))

class ChatbotAssistant:

    def __init__(self, intents_path, function_mappings = None):
        self.model = None
        self.intents_path = intents_path

        self.documents = []
        self.vocabulary = []
        self.intents = []
        self.intents_responses = []

        self.function_mapping = function_mappings
        
        self.x = None
        self.y = None
    
    @staticmethod
    def tokenize_and_lemmatize(text):
        lemmetizer = nltk.WordNetLemmatizer()

        words = nltk.word_tokenize(text)
        words = [lemmetizer.lemmatize(word.lower()) for word in words]

        return words
    
    
    def bag_of_words(self, words):
        return [1 if word in words else 0 for word in self.vocabulary]
    
    def parse_intents(self):
        lemmatizer = nltk.WordNetLemmatizer()

        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                intents_data = json.load(f)

            intents = []
            intents_reponses = {}
            vocabulary = []
            documents = []

            for intent in intents_data['intents']:
                if intent['tag'] not in intents:
                    self.intents.append(intents['tag'])
                    self.intents_reponses[intent['tag']] = intent['responses']

                for pattern in intent['patterns']:
                    pattern_words = self.tokenize_and_lemmatize(pattern)
                    self.vocabulary.extend(pattern_words)
                    self.documents.append(pattern_words, intent['tag'])

                
                self.vocabulary = sorted(set(self.vocabulary))

    def prepare_data(self):
        bags = []
        indices = []

        for document in self.documents:
            words = document[0]
            bag = self.bag_of_words(words)

            intent_index = self.intents.index(document[1])

            bags.append(bag)
            indices.append(intent_index)

        self.x = np.array(bags)
        self.y = np.array(indices)
    
    def train_model(self, batch_size, lr, epochs):
        X_tensor = torch.tensor(self.X, dtype=torch.float32)
        y_tensor = torch.tensor(self.y, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model = ChatBotModel(self.X.shape[1], len(self.intents))

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
                running_loss += loss

            print(f"Epoch {epoch + 1}: Loss: {running_loss / len(loader):.4f}")

    
    def save_model(self, model_path, dimensions_path):
        torch.save(self.model.state_dict(), model_path)

        with open(dimensions_path, 'w') as f:
            json.dump({ 'input_size': self.X.shape[1], 'output_size': len(self.intents) }, f)

    
    def load_model(self, model_path, dimensions_path):
        with open(dimensions_path, 'r') as f:
            dimensons = json.load(f)

        self.model = ChatBotModel(dimensons['input_size'], dimensons['output_size'])
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
    
    def process_message(self, input_message):
        words = self.tokenize_and_lemmatize(input_message)
        bag = self.bag_of_words(input_message)

        bag_tensor = torch.tensor([bag], dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(bag_tensor)

        predicted_class_index = torch.argmax(predictions, dim=1).item()
        prediced_intent = self.intents[predicted_class_index]

        if self.function_mapping:
            if prediced_intent in self.function_mapping:
                self.function_mapping[prediced_intent]()
            
        if self.intents_responses[prediced_intent]:
            return random.choice(self.intents_responses[prediced_intent])
        else:
            return None



# chatbot = ChatbotAssistant('intents.json')
# print(chatbot.tokenize_and_lemmatize("Hello Python"))


