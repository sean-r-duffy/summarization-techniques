from datasets import load_dataset, DatasetDict
import torch
# from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
import nltk
# nltk.download('punkt_tab')
# import numpy as np
# from nltk.tokenize import sent_tokenize
# from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader #, Dataset
import torch.nn as nn
from functions import *
import json
from tqdm import tqdm
print("PACKAGES SUCCESS")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# load data (arxiv)
# ds = load_dataset("ccdv/arxiv-summarization", "section")
# ds_short = DatasetDict({"train": ds["train"].select(range(100)),
#             "validation": ds["validation"].select(range(1)),
#             "test": ds["test"].select(range(1))})

# load data pubmed
ds = load_dataset("ccdv/pubmed-summarization", "section")


# Create Dataset and DataLoader
train_dataset = ArxivSummarizationDataset(ds["train"])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Create FFNN and define optimizer, loss, epochs
scoring_model = RelevanceScoringModel().to(device)
optimizer = torch.optim.Adam(scoring_model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()
num_epochs = 5

# Store training results
training_dict = {}

# Training Loop
print("TRAINING STARTING")
for epoch in range(num_epochs):
    total_loss = 0.0
    scoring_model.train()

    for batch in tqdm(train_loader):
        sentence_embeddings = batch["sentence_embeddings"].to(device)
        cosine_labels = batch["cosine_labels"].to(device)

        # Forward pass
        predictions = scoring_model(sentence_embeddings).squeeze(-1)
        loss = criterion(predictions, cosine_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss}")
    # print("Epoch: ", epoch, "Loss: ", total_loss)
    training_dict[epoch+1] = round(total_loss,4)

# Results and model storing 
with open("training_results.json", "w") as outfile: 
    json.dump(training_dict, outfile)

torch.save(scoring_model.state_dict(), "pubmed_relevance_scoring_model.pt")
