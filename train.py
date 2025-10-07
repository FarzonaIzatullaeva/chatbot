from model import NeuralNet
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch
from nltk_utils import tokenize
import json
with open('intents.json', 'r') as f:
    intents = json.load(f)
from nltk_utils import stem
from nltk_utils import bag_of_words

import numpy as np

all_words = []

tags = []

xy = []

# get the tags and the words from the intents.json and locate them in the all_words and tags arrays and the two touple of the word and tag in the xy array.
for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))


ignore_words = ['?', '!', '.', ',']

# stem the words to their base from the all_words array ignoring the things in the ignore_words array
all_words = [stem(w) for w in all_words if w not in ignore_words]
print(all_words)

# sort the words and tags in ascending order with removing duplicated
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(tags)

x_train = []
y_train = []

for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    x_train.append(bag)

    label = tags.index(tag)

    y_train.append(label)  # CrossEntropyLoss

x_train = np.array(x_train)
y_train = np.array(y_train)

# Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.001
num_epochs = 1000


print(input_size, len(all_words))
print(output_size)
print(output_size, tags)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    # dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(
    dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size)

# loss and optimizer
# criterion will measure how far the model's predictions are from the true labels.
criterion = nn.CrossEntropyLoss()

# optimizer tells PyTorch how to update the models' weights to reduce the loss
# Adam is a otimization algorithm
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# if the data set has lets say 200 samples those 200 samples take 1000 epochs or 1000 turns for the data set to pass through the model.
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        # forward
        # predict the score for each tag
        output = model(words)
        # calculate how wrong the predictions are
        loss = criterion(output, labels)

        # backward and optimizer step
        # clear gradients from previous batch
        optimizer.zero_grad()
        # compute the gradient of the loss
        loss.backward()
        # update weights using the gradient (Adam optimizer rule)
        optimizer.step()

    # for every 100 epochs print the current epoch number and the loss.
    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')


data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. File saved to {FILE}')
