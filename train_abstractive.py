from abstractive.models import AbstractiveSummarizer

if __name__ == '__main__':
    dataset = 'ccdv/pubmed-summarization'
    name = 'pubmed'
    summarizer = AbstractiveSummarizer(name=name)
    summarizer.train(dataset=dataset, checkpoint_dir=f'models/abstractive/{name}')
    summarizer.evaluate(dataset='ccdv/arxiv-summarization', dataset_name='arxiv')
    summarizer.evaluate(dataset='ccdv/pubmed-summarization', dataset_name='pubmed')
    summarizer.evaluate(dataset='ccdv/govreport-summarization', dataset_name='govreport', text_col='report',
                        target_col='summary')

    dataset = 'ccdv/arxiv-summarization'
    name = 'arxiv'
    summarizer = AbstractiveSummarizer(name=name)
    summarizer.train(dataset=dataset, checkpoint_dir=f'models/abstractive/{name}')
    summarizer.evaluate(dataset='ccdv/arxiv-summarization', dataset_name='arxiv')
    summarizer.evaluate(dataset='ccdv/pubmed-summarization', dataset_name='pubmed')
    summarizer.evaluate(dataset='ccdv/govreport-summarization', dataset_name='govreport', text_col='report', target_col='summary')

    dataset = 'ccdv/govreport-summarization'
    name = 'govreport'
    summarizer = AbstractiveSummarizer(name=name)
    summarizer.train(dataset=dataset, checkpoint_dir=f'models/abstractive/{name}', input_col='report', target_col='summary')
    summarizer.evaluate(dataset='ccdv/arxiv-summarization', dataset_name='arxiv')
    summarizer.evaluate(dataset='ccdv/pubmed-summarization', dataset_name='pubmed')
    summarizer.evaluate(dataset='ccdv/govreport-summarization', dataset_name='govreport', text_col='report', target_col='summary')