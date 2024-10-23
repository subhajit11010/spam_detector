# import urllib.request
# import zipfile
# import os
# import pathlib
import torch
# import torch.nn as nn
# import tiktoken
# import matplotlib.pyplot as plt
# import chardet
# from pathlib import Path
from gptmodel_with_pretrained_weights import model_configs, GPTModel, load_weights_into_gpt
# from gpt_download import download_and_load_gpt2

# tokenizer = tiktoken.get_encoding("gpt2")
BASE_CONFIG = {
 "vocab_size": 50257,
 "context_length": 1024,
 "drop_rate": 0.0,
 "qkv_bias": True,
}
# url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
# zip_path = "sms_spam_collection.zip"
# extracted_path = "sms_spam_collection"
# data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"

# def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path):

#   if data_file_path.exists():
#     print(f"{data_file_path} already exists. Skipping download and extraction.")
#     return
#   with urllib.request.urlopen(url) as response:
#     with open(zip_path, "wb") as out_file:
#       out_file.write(response.read())
#   with zipfile.ZipFile(zip_path, "r") as zip_ref:
#     zip_ref.extractall(extracted_path)
#   original_file_path = Path(extracted_path) / "SMSSpamCollection"
#   os.rename(original_file_path, data_file_path)
#   print(f"File downloaded and saved as {data_file_path}")
# download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path)


# import pandas as pd
# with open('./spam.csv', 'rb') as f:
#     result = chardet.detect(f.read())
# encoding = result['encoding']
# df = pd.read_csv('./spam.csv', header = None, names=["Label", "Text"], encoding=encoding, usecols=[0, 1])
# # print(df.head())
# def create_balanced_dataset(df):
#   num_spam = df[df["Label"] == "spam"].shape[0]
#   ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state = 123)
#   balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
#   return balanced_df
# balanced_df = create_balanced_dataset(df)
# balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

# def random_split(df, train_frac, validation_frac):

#  df = df.sample(frac=1, random_state=123).reset_index(drop=True)
#  train_end = int(len(df) * train_frac)
#  validation_end = train_end + int(len(df) * validation_frac)

#  train_df = df[:train_end]
#  validation_df = df[train_end:validation_end]
#  test_df = df[validation_end:]
#  return train_df, validation_df, test_df
# train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)

# train_df.to_csv("./sample_data/train.csv", index = None)
# validation_df.to_csv("./sample_data/validation.csv", index = None)
# test_df.to_csv("./sample_data/test.csv", index = None)

# import torch
# from torch.utils.data import Dataset
# class SpamDataset(Dataset):
#   def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
#     self.data = pd.read_csv(csv_file)
#     self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
#     if max_length is None:
#       self.max_length = self._longest_encoded_length()
#     else:
#       self.max_length = max_length
#       self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]
#     self.encoded_texts = [encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts]

#   def __getitem__(self, index):
#     encoded = self.encoded_texts[index]
#     label = self.data.iloc[index]["Label"]
#     return (torch.tensor(encoded, dtype=torch.long),torch.tensor(label, dtype=torch.long))

#   def __len__(self):
#     return len(self.data)

#   def _longest_encoded_length(self):
#     max_length = 0
#     for encoded_text in self.encoded_texts:
#       encoded_length = len(encoded_text)
#       if encoded_length > max_length:
#         max_length = encoded_length
#     return max_length

# train_dataset = SpamDataset("./sample_data/train.csv", tokenizer, None)
# # print(train_dataset.max_length)
# val_dataset = SpamDataset("./sample_data/validation.csv", tokenizer, max_length = train_dataset.max_length)
# test_dataset = SpamDataset("./sample_data/test.csv", tokenizer, max_length = train_dataset.max_length)

# from torch.utils.data import DataLoader
# num_workers = 0
# batch_size = 8
# torch.manual_seed(123)
# train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, drop_last = True)
# val_loader = DataLoader(dataset = val_dataset, batch_size = batch_size, num_workers = num_workers, drop_last = False)
# test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, num_workers = num_workers, drop_last = False)

# CHOOSE_MODEL = "gpt2-small (124M)"
# BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
# # settings, params = download_and_load_gpt2(
# #  model_size=model_size, models_dir="gpt2"
# # )
# model = GPTModel(BASE_CONFIG)
# # load_weights_into_gpt(model, params)
# for param in model.parameters():
#   param.requires_grad = False
# torch.manual_seed(123)
# num_classes = 2
# model.out_head = nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
# for param in model.trf_blocks[-1].parameters():
#   requires_grad = True
# for param in model.final_norm.parameters():
#   requires_grad = True

# def calc_accuracy_loader(data_loader, model, device, num_batches=None):
#   model.eval()
#   correct_predictions, num_examples = 0 , 0
#   if num_batches is None:
#     num_batches = len(data_loader)
#   else:
#     num_batches = min(num_batches, len(data_loader))
#   for i, (input_batch, target_batch) in enumerate(data_loader):
#     if i < num_batches:
#       input_batch = input_batch.to(device)
#       target_batch = target_batch.to(device)
#       with torch.no_grad():
#         logits = model(input_batch)[:,-1,:]
#       predicted_labels = torch.argmax(logits, dim=-1)
#         # print(predicted_labels)
#       num_examples += predicted_labels.shape[0]
#       correct_predictions += (predicted_labels == target_batch).sum().item()
#     else:
#       break
#   return correct_predictions / num_examples

# def calc_loss_batch(input_batch, target_batch, model, device):
#     input_batch = input_batch.to(device)
#     target_batch = target_batch.to(device)
#     logits = model(input_batch)[:,-1,:]
#     loss = torch.nn.functional.cross_entropy(logits, target_batch)
#     return loss

# def calc_loss_loader(data_loader, model, device, num_batches = None):
#     total_loss = 0.0
#     if len(data_loader) == 0:
#         return float("nan")
#     elif num_batches is None:
#         num_batches = len(data_loader)
#     else:
#         num_batches = min(num_batches, len(data_loader))
#     for i, (input_batch, target_batch) in enumerate(data_loader):
#         if i < num_batches:
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             total_loss += loss.item()
#         else:
#             break
#     return total_loss / num_batches
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
#     train_losses, val_losses, train_accs, val_accs = [], [], [], []
#     examples_seen, global_step = 0, -1

#     for epoch in range(num_epochs):
#         model.train()
#         for input_batch, target_batch in train_loader:
#             optimizer.zero_grad()
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             loss.backward()
#             optimizer.step()
#             examples_seen += input_batch.shape[0]
#             global_step += 1

#             if global_step % eval_freq == 0:
#                 train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
#                 train_losses.append(train_loss)
#                 val_losses.append(val_loss)
#                 print(f"Ep {epoch+1} (Step {global_step:06d}): "
#                       f"Train loss {train_loss:.3f}, "
#                       f"Val loss {val_loss:.3f}"
#                 )
#         train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches = eval_iter)
#         val_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches = eval_iter)
#         print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
#         print(f"Validation accuracy: {val_accuracy*100:.2f}%")
#         train_accs.append(train_accuracy)
#         val_accs.append(val_accuracy)
#     return train_losses, val_losses, train_accs, val_accs, examples_seen

# def evaluate_model(model, train_loader, val_loader, device, eval_iter):
#     model.eval()
#     with torch.no_grad():
#         train_loss = calc_loss_loader(train_loader, model, device, num_batches = eval_iter)
#         val_loss = calc_loss_loader(val_loader, model, device, num_batches = eval_iter)
#     model.train()
#     return train_loss, val_loss

# import time
# model.to(device)
# # start_time = time.time()
# # torch.manual_seed(123)
# # optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
# # num_epochs = 5
# # train_losses, val_losses, train_accs, val_accs, examples_seen = \
# #     train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs=num_epochs,eval_freq=50, eval_iter=5)
# # end_time = time.time()
# # execution_time_minutes = (end_time - start_time) / 60
# # print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# def plot_values(
#   epochs_seen, examples_seen, train_values, val_values,
#   label="loss"):
#   fig, ax1 = plt.subplots(figsize=(5, 3))

#   ax1.plot(epochs_seen, train_values, label=f"Training {label}")
#   ax1.plot(
#   epochs_seen, val_values, linestyle="-.",
#   label=f"Validation {label}"
#   )
#   ax1.set_xlabel("Epochs")
#   ax1.set_ylabel(label.capitalize())
#   ax1.legend()

#   ax2 = ax1.twiny()
#   ax2.plot(examples_seen, train_values, alpha=0)
#   ax2.set_xlabel("Examples seen")
#   fig.tight_layout()
#   plt.savefig(f"{label}-plot.pdf")
#   plt.show()
# # epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# # examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
# # plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

# # epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
# # examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))
# # plot_values(
# #  epochs_tensor, examples_seen_tensor, train_accs, val_accs,
# #  label="accuracy"
# # )

# train_accuracy = calc_accuracy_loader(train_loader, model, device)
# val_accuracy = calc_accuracy_loader(val_loader, model, device)
# test_accuracy = calc_accuracy_loader(test_loader, model, device)
# # print(f"Training accuracy: {train_accuracy*100:.2f}%")
# # print(f"Validation accuracy: {val_accuracy*100:.2f}%")
# # print(f"Test accuracy: {test_accuracy*100:.2f}%")

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id = 50256):
    model.eval()
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]
    input_ids = input_ids[:min(max_length, supported_context_length)]
    input_ids += [pad_token_id] * (max_length - len(input_ids))
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)[:,-1,:]
    predicted_label = torch.argmax(logits, dim=-1).item()
    return "spam" if predicted_label == 1 else "not spam"

# text_1 = (
#  "Congratulations! You've won a brand new iPhone!"
#  "Click here to claim your prize: [www.fakeprize.com]"

# )
# # print(classify_review(
# #  text_1, model, tokenizer, device, max_length=train_dataset.max_length
# # ))

# # torch.save({"model_state_dict": model.state_dict(),
# #             "optimizer_state_dict": optimizer.state_dict()}, "./review_classifier.pth")