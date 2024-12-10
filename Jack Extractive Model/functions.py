from datasets import load_dataset, DatasetDict
import torch
from transformers import DistilBertModel, DistilBertTokenizer
import nltk
import numpy as np
from nltk.tokenize import sent_tokenize

from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
print("PACKAGES SUCCESS")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# distilBERT
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
embedding_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

# reworked to handle batching
def get_sentence_embeddings(sentences, batch_size = 8):
    if not sentences:
        return torch.zeros((1, 768), device= device)
    
    embeddings = []

    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size] # get batch and pass into BERT model
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = embedding_model(**inputs)   
        embeddings.append(outputs.last_hidden_state[:, 0, :]) # add CLS tokens to list 

        return torch.cat(embeddings, dim= 0).to(device) # concat tensors together 
    

def calc_cosine_sim(body_embeddings, summary_embedding, threshold=0.75):
    # normalize body and summary embeddings
    body_embeddings = body_embeddings / body_embeddings.norm(dim= 1, keepdim= True)
    summary_embedding = summary_embedding / summary_embedding.norm(dim= 1, keepdim= True)

    # matrix multiplication to get cosine matrix
    sim_matrix = torch.mm(body_embeddings, summary_embedding.T)
    labels = (sim_matrix.squeeze(-1) >= threshold).to(torch.float32) # check if above threshold
    return labels

# Class for dataloader 
class ArxivSummarizationDataset(Dataset):
    def __init__(self, dataset, max_sentences=50): # max sentences limits length
        self.dataset = dataset
        self.max_sentences = max_sentences

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # for arxiv and pubmed data
        sentences = self.dataset[idx]["article"][:self.max_sentences]
        summary = self.dataset[idx]["abstract"]

        # # for gov data
        # sentences = self.dataset[idx]["report"][:self.max_sentences]
        # summary = self.dataset[idx]["summary"]

        # Embed sentences and summary
        sentence_embeddings = get_sentence_embeddings(sentences).to(device)
        summary_embedding = get_sentence_embeddings([summary]).mean(dim=0, keepdim=True)

        # Calculate cosine similarity labels
        labels = calc_cosine_sim(sentence_embeddings, summary_embedding).to(device)

        return {
            "sentence_embeddings": sentence_embeddings.to(device),
            "cosine_labels": labels.to(device)}
    



class RelevanceScoringModel(nn.Module):
    def __init__(self, input_dim=768):  # 768 for BERT's embedding size
        super(RelevanceScoringModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)  # Output relevance score for each sentence

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
print("Functions loaded")