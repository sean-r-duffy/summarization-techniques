from abstractive.models import AbstractiveSummarizer
import json

FROM_HF = False

if FROM_HF:
    with open('../models/abstractive/hf_names.json', 'r') as f:
        data = json.load(f)
    arxiv_model = data['arxiv']
    pubmed_model = data['pubmed']
    govreport_model = data['govreport']
else:
    arxiv_model = '../models/abstractive/arxiv_final'
    pubmed_model = '../models/abstractive/pubmed_final'
    govreport_model = '../models/abstractive/govreport_final'

if __name__ == '__main__':
    summarizer = AbstractiveSummarizer(model_path=pubmed_model, name='pubmed')
    summarizer.evaluate(dataset='ccdv/arxiv-summarization', dataset_name='arxiv')
    summarizer.evaluate(dataset='ccdv/pubmed-summarization', dataset_name='pubmed')
    summarizer.evaluate(dataset='ccdv/govreport-summarization', dataset_name='govreport', text_col='report',
                        target_col='summary')

    summarizer = AbstractiveSummarizer(model_path=arxiv_model, name='arxiv')
    summarizer.evaluate(dataset='ccdv/arxiv-summarization', dataset_name='arxiv')
    summarizer.evaluate(dataset='ccdv/pubmed-summarization', dataset_name='pubmed')
    summarizer.evaluate(dataset='ccdv/govreport-summarization', dataset_name='govreport', text_col='report', target_col='summary')

    summarizer = AbstractiveSummarizer(model_path=govreport_model, name='govreport')
    summarizer.evaluate(dataset='ccdv/arxiv-summarization', dataset_name='arxiv')
    summarizer.evaluate(dataset='ccdv/pubmed-summarization', dataset_name='pubmed')
    summarizer.evaluate(dataset='ccdv/govreport-summarization', dataset_name='govreport', text_col='report', target_col='summary')