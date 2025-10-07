from flask import Flask, render_template, request, jsonify
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Load the chatbot model and data
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with open('intents.json', 'r') as f:
    intents = json.load(f)
FILE = 'data.pth'
data = torch.load(FILE)
# (rest of the data loading code from chat.py)
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]
model = NeuralNet(data["input_size"], data["hidden_size"],
                  data["output_size"]).to(device)
model.load_state_dict(model_state)
model.eval()

app = Flask(__name__)


@app.route("/")
def home():
    # Renders the HTML file with the chat interface
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    # Get the user message from the frontend
    user_message = request.get_json().get("message")

    # Run the chatbot logic from chat.py
    sentence = tokenize(user_message)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x).to(device)
    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    response_text = "I do not understand..."
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                response_text = random.choice(intent['responses'])

    # Return the chatbot's response as JSON
    return jsonify({"answer": response_text})


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
