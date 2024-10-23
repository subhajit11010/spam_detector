from flask import Flask, request, jsonify
import torch
import tiktoken
from gptmodel_with_pretrained_weights import GPTModel, model_configs
from spam_detector import classify_review, BASE_CONFIG

app = Flask(__name__)
CHOOSE_MODEL = "gpt2-small (124M)"
tokenizer = tiktoken.get_encoding('gpt2')
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(BASE_CONFIG)
checkpoint = torch.load('./review_classifier.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

@app.route('/predict', methods=["POST"])
def predict():
    data = request.json
    text = data['text']

    label = classify_review(text, model, tokenizer, device)
    return jsonify({"label": label})

if __name__ == "__main__":
    app.run(debug=True)