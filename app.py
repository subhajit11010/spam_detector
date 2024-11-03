from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import os
import tiktoken
from gptmodel_with_pretrained_weights import GPTModel, model_configs
from spam_detector import classify_review, BASE_CONFIG

app = Flask(__name__)
CHOOSE_MODEL = "gpt2-small (124M)"
tokenizer = tiktoken.get_encoding('gpt2')
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(BASE_CONFIG)
num_classes = 2
model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
checkpoint = torch.load('./review_classifier.pth', map_location=device, weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    data = request.get_json()
    print(data)
    if not data or 'text' not in data:
        return jsonify({"error": "Invalid input, 'text' field is required"}), 400
    text = data['text']

    prediction = classify_review(text, model, tokenizer, device, max_length=120)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)