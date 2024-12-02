from datasets import load_dataset, DatasetDict
import torch
import nltk
from nltk.tokenize import sent_tokenize
from functions import *
from rouge_score import rouge_scorer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

test_dataset = load_dataset("ccdv/arxiv-summarization", split= "test")
validation_dataset = load_dataset("ccdv/arxiv-summarization", split= "validation")

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

model = RelevanceScoringModel()
model.load_state_dict(torch.load("relevance_scoring_model.pt", weights_only= True))
model.eval()

# summary_list = []
# for i in range(len(test_dataset)):
#     gen_sum = generate_summary(model, test_dataset[i]["article"])
#     summary_list.append([test_dataset[i]["abstract"], gen_sum])

# print("SUMMARY LIST LENGTH = ", len(summary_list))
# print("DONE")



# # Load a test example
# test_example = test_dataset[0]
# test_article = test_example["article"]
# test_abstract = test_example["abstract"]  

# # Generate summary
# generated_summary = generate_summary(model, test_article)

# # Print the results
# print("Original Abstract:")
# print(test_abstract)
# print("\nGenerated Summary:")
# print(generated_summary)




