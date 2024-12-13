from abstractive.models import AbstractiveSummarizer

if __name__ == '__main__':
    dataset = 'ccdv/pubmed-summarization'
    name = 'pubmed'
    summarizer = AbstractiveSummarizer(name=name)
    summarizer.train(dataset=dataset, checkpoint_dir=f'models/abstractive/{name}')

    dataset = 'ccdv/arxiv-summarization'
    name = 'arxiv'
    summarizer = AbstractiveSummarizer(name=name)
    summarizer.train(dataset=dataset, checkpoint_dir=f'models/abstractive/{name}')

    dataset = 'ccdv/govreport-summarization'
    name = 'govreport'
    summarizer = AbstractiveSummarizer(name=name)
    summarizer.train(dataset=dataset, checkpoint_dir=f'models/abstractive/{name}', input_col='report', target_col='summary')