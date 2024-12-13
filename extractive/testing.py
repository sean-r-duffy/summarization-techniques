from datasets import load_dataset, DatasetDict
import torch
import nltk
from nltk.tokenize import sent_tokenize
from functions import *
import evaluate
from rouge_score import rouge_scorer
from tqdm import tqdm
import json

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# scorer
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)

# testing setup
model_list = [("arxiv_relevance_scoring_model.pt", "arxiv_model"), ("pubmed_relevance_scoring_model.pt", "pubmed_model"), ("govrep_relevance_scoring_model.pt", "govrep_model")]

arxiv_test_dataset = load_dataset("ccdv/arxiv-summarization", split= "test")
pubmed_test_dataset = load_dataset("ccdv/pubmed-summarization", split= "test")
govrep_test_dataset = load_dataset("ccdv/govreport-summarization", split= "test")

# (dataset name, structure of article and summary, string name)
dataset_list = [(arxiv_test_dataset, 0, "arxiv"), (pubmed_test_dataset, 0, "pubmed"), (govrep_test_dataset, 1, "govrep")]

# test_indexes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

# arxiv_examples = []
# pubmed_examples = []
# govrep_examples = []

# for i in test_indexes:
#     arxiv_examples.append(arxiv_test_dataset[i])
#     pubmed_examples.append(pubmed_test_dataset[i])
#     govrep_examples.append(govrep_test_dataset[i])

# dataset_list = [(arxiv_examples, 0, "arxiv"), (pubmed_examples, 0, "pubmed"), (govrep_examples, 1, "govrep")] 
    

def generate_summary(model, article, threshold=0.5, max_sentences=50, summary_length=5):
    # Tokenize the article into sentences
    sentences = sent_tokenize(article)[:max_sentences]

    # Embed the sentences
    sentence_embeddings = get_sentence_embeddings(sentences)

    # Predict relevance scores
    model.eval() 
    with torch.no_grad():
        logits = model(sentence_embeddings)

    # Apply sigmoid to get relevance scores
    relevance_scores = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()  

    # Rank sentences by relevance scores
    # Get indices of the top `summary_length` sentences
    top_indices = np.argsort(relevance_scores)[-summary_length:][::-1]  

    # Select the top sentences
    relevant_sentences = [sentences[i] for i in top_indices]

    return relevant_sentences



# output dictionary
data_dict = {}

print("inference starting")
for model_string, model_name in tqdm(model_list): # iterate through each model
    # load in model
    model = RelevanceScoringModel()
    model.load_state_dict(torch.load(model_string, weights_only= True))
    model.to(device) # to gpu
    model.eval()
    data_dict[model_name] = {} # create first dictionary 
    for dataset in dataset_list: # iterate through datasets and store name
        dataset_name = dataset[2]
        data_dict[model_name][dataset_name] = {} # next dictionary 
        if dataset[1] == 0: # pubmed and arxiv
            for i in range(len(dataset[0])):
                # get summary and article then generate summary 
                example = dataset[0][i]
                article = example["article"]
                abstract = example["abstract"]

                gen_abstract = generate_summary(model, article)
                gen_abstract = " ".join(gen_abstract)

                # score results and store 
                score_results = scorer.score(abstract, gen_abstract)
                data_dict[model_name][dataset_name][i] = score_results
        else:
            for i in range(len(dataset[0])): # govreport 
                example = dataset[0][i]
                article = example["report"]
                abstract = example["summary"]

                gen_abstract = generate_summary(model, article)
                gen_abstract = " ".join(gen_abstract)

                score_results = scorer.score(abstract, gen_abstract)
                data_dict[model_name][dataset_name][i] = score_results

# save dictionary 
with open("ROUGE_scores_full.json", "w") as outfile: 
    json.dump(data_dict, outfile)






