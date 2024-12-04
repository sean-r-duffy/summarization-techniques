from datasets import load_dataset, DatasetDict
import torch
import nltk
from nltk.tokenize import sent_tokenize
from functions import *
import evaluate
from rouge_score import rouge_scorer
from tqdm import tqdm
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=False)
model_list = [("arxiv_relevance_scoring_model.pt", "arxiv_model"), ("pubmed_relevance_scoring_model.pt", "pubmed_model"), ("govrep_relevance_scoring_model.pt", "govrep_model")]

arxiv_test_dataset = load_dataset("ccdv/arxiv-summarization", split= "test")
pubmed_test_dataset = load_dataset("ccdv/pubmed-summarization", split= "test")
govrep_test_dataset = load_dataset("ccdv/pubmed-summarization", split= "test")

test_indexes = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

arxiv_examples = []
pubmed_examples = []
govrep_examples = []

for i in test_indexes:
    arxiv_examples.append(arxiv_test_dataset[i])
    pubmed_examples.append(pubmed_test_dataset[i])
    govrep_examples.append(govrep_test_dataset[i])

dataset_list = [(arxiv_examples, 0, "arxiv"), (pubmed_examples, 0, "pubmed"), (govrep_examples, 1, "govrep")] 
    

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

# model = RelevanceScoringModel()
# model.load_state_dict(torch.load("relevance_scoring_model.pt", weights_only= True))
# model.eval()

data_dict = {}

for model_string, model_name in tqdm(model_list):
    model = RelevanceScoringModel()
    model.load_state_dict(torch.load(model_string, weights_only= True))
    model.to(device)
    model.eval()
    for dataset in dataset_list:
        if dataset[1] == 0:
            for i in range(len(dataset_list)):
                example = dataset[0][i]
                article = example["article"]
                abstract = example["abstract"]

                gen_abstract = generate_summary(model, article)
                gen_abstract = " ".join(gen_abstract)

                score_results = scorer.score(abstract, gen_abstract)
                data_dict[model_name][dataset[2]] = score_results
        else:
            for i in range(len(dataset_list)):
                example = dataset[0][i]
                article = example["report"]
                abstract = example["summary"]

                gen_abstract = generate_summary(model, article)
                gen_abstract = " ".join(gen_abstract)

                score_results = scorer.score(abstract, gen_abstract)
                data_dict[model_name][dataset[2]] = score_results

with open("ROUGE_scores.json", "w") as outfile: 
    json.dump(data_dict, outfile)



# Load a test example
# test_example = arxiv_test_dataset[0]
# test_article = test_example["article"]
# test_abstract = test_example["abstract"]  

# # Generate summary
# generated_summary = generate_summary(model, test_article)
# generated_summary = " ".join(generated_summary)


# score_results = scorer.score(test_abstract, generated_summary)
# print(score_results)


# # Print the results
# print("Original Abstract:")
# print(test_abstract)
# print("\nGenerated Summary:")
# print(generated_summary)




