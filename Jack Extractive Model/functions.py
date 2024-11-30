from datasets import load_dataset, DatasetDict
import torch
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
import nltk
nltk.download('punkt_tab')
import numpy as np
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
print("PACKAGES SUCCESS")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load BERT model and tokenizer for sentence embeddings
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
embedding_model = BertModel.from_pretrained("bert-base-uncased")

def get_sentence_embeddings(sentences):
    inputs = tokenizer(sentences, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = embedding_model(**inputs)
    return outputs.last_hidden_state[:, 0, :]  # [CLS] token embeddings


def calc_cosine_sim(body_embeddings, summary_embedding, threshold=0.75):
    sim_matrix = cosine_similarity(body_embeddings.numpy(), summary_embedding.numpy())
    labels = (sim_matrix.max(axis=1) >= threshold).astype(int)
    return labels


class ArxivSummarizationDataset(Dataset):
    def __init__(self, dataset, max_sentences=50):
        self.dataset = dataset
        self.max_sentences = max_sentences

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sentences = self.dataset[idx]["article"][:self.max_sentences]
        summary = self.dataset[idx]["abstract"]

        # Embed sentences and summary
        sentence_embeddings = get_sentence_embeddings(sentences)
        summary_embedding = get_sentence_embeddings([summary]).mean(dim=0, keepdim=True)

        # Calculate cosine similarity labels
        labels = calc_cosine_sim(sentence_embeddings, summary_embedding)

        return {
            "sentence_embeddings": sentence_embeddings,
            "cosine_labels": torch.tensor(labels, dtype=torch.float32)
        }
    

    
class RelevanceScoringModel(nn.Module):
    def __init__(self, input_dim=768):  # 768 is BERT's embedding size
        super(RelevanceScoringModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)  # Output relevance score for each sentence

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x