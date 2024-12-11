from datasets import load_dataset, DatasetDict
import torch
import nltk
from torch.utils.data import DataLoader
import torch.nn as nn
from functions import *
import json
from tqdm import tqdm
print("PACKAGES SUCCESS")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# # load data (arxiv)
# print("USING ARXIV DATA")
# ds = load_dataset("ccdv/arxiv-summarization", "section")

# load data pubmed
print("USING PUBMED")
ds = load_dataset("ccdv/pubmed-summarization", "section")

# # load data govreport (MUST CHANGE ARTICLE AND ABSTRACT TO FIT)
# print("gov data")
# ds = load_dataset("ccdv/govreport-summarization")

# # dataset for testing
# ds_short = DatasetDict({"train": ds["train"].select(range(100)),
#             "validation": ds["validation"].select(range(1)),
#             "test": ds["test"].select(range(1))})


# Create Dataset and DataLoader
train_dataset = ArxivSummarizationDataset(ds["train"])
train_loader = DataLoader(train_dataset, batch_size= 32, shuffle= True)

val_dataset = ArxivSummarizationDataset(ds["validation"])
val_loader = DataLoader(val_dataset, batch_size= 32, shuffle= False)

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

    for batch in tqdm(train_loader, desc= f"Training Epoch: {epoch}"):
        # embed sentences
        sentence_embeddings = batch["sentence_embeddings"].to(device)
        cosine_labels = batch["cosine_labels"].to(device)

        # Forward pass
        predictions = scoring_model(sentence_embeddings).squeeze(-1)
        loss = criterion(predictions, cosine_labels)

        # Backward and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Validation
    scoring_model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc= f"Validation Epoch: {epoch}"):
            # embed sentences
            sentence_embeddings = batch["sentence_embeddings"].to(device)
            cosine_labels = batch["cosine_labels"].to(device)

            # forward pass
            predictions = scoring_model(sentence_embeddings).squeeze(-1)

            # store loss
            loss = criterion(predictions, cosine_labels)
            val_loss += loss.item()
    
    # average losses
    avg_train = total_loss / len(train_loader)
    avg_val = val_loss / len(val_loader)

    print(f"Epoch {epoch}:   Training Loss: {avg_train}   Validation Loss: {avg_val}")

    training_dict[epoch+1] = {"train_loss": round(avg_train, 4),
                              "validation_loss": round(avg_val, 4)}

# Results and model storing 
with open("pubmed_training_results.json", "w") as outfile: 
    json.dump(training_dict, outfile)

torch.save(scoring_model.state_dict(), "pubmed_relevance_scoring_model.pt")
