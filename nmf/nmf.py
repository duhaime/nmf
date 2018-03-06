from __future__ import division, print_function
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from sklearn import decomposition
import sys, glob, json, codecs, os, argparse
import numpy as np

class NMF:
  def __init__(self, files=None, encoding='utf8', max_files=None,
      topics=20, iters=100, words=10, write_output=False, min_count=2,
      max_freq=0.95):
    self.files = files
    self.encoding = encoding
    self.max_files = max_files
    self.topics = topics
    self.iters = iters
    self.words = words
    self.write_output = write_output
    self.min_count = min_count
    self.max_freq = max_freq

    # build corpus
    self.infiles = self.get_infiles(self.files, self.max_files)
    self.corpus = self.build_corpus(self.infiles, self.encoding)

    # build tdm
    self.vectorizer = self.get_vectorizer(self.topics, self.words)
    self.tdm = self.get_tdm(self.vectorizer, self.corpus)

    # factor tdm matrix
    self.feature_names = self.vectorizer.get_feature_names()
    self.nmf = self.build_nmf()

    # get topics by documents and topics by terms
    self.documents_by_topics = self.get_documents_by_topics()
    self.topics_by_terms = self.nmf.components_

    # get topics by documents and topics by terms
    self.docs_to_topics = self.get_doc_to_topics()
    self.topics_to_words = self.get_topic_to_words()

    # write the results
    if self.write_output:
      self.write_results()

  def write_json(self, filename, obj):
    if not os.path.exists('results'):
      os.makedirs('results')
    with open(os.path.join('results', filename), 'w') as out:
      json.dump(obj, out)

  def read_file(self, path, encoding):
    with codecs.open(path, 'r', encoding) as f:
      return f.read()

  def build_corpus(self, infiles, encoding):
    for i in infiles:
      yield self.read_file(i, encoding)

  def get_infiles(self, root_dir, max_files):
    '''
    Recursively search for all files within `root_dir` and return
    up to max_files of the found files
    '''
    infiles = []
    keep_matching = True
    for root, dirnames, filenames in os.walk(root_dir):
      if keep_matching:
        for filename in filenames:
          infiles.append(os.path.join(root, filename))
          if max_files and len(infiles) >= max_files:
            keep_matching = False
            break
    return infiles

  def get_vectorizer(self, topics, n_words):
    '''
    Return a TFIDF Vectorizer for building the input TDM matrix
    '''
    return TfidfVectorizer(
      input='content',
      stop_words='english',
      max_df=self.max_freq,
      min_df=self.min_count,
      max_features=topics * n_words * 1000
    )

  def get_tdm(self, vectorizer, corpus):
    '''
    Return a TDM to factor
    '''
    return vectorizer.fit_transform(corpus)

  def build_nmf(self):
    return decomposition.NMF(n_components=self.topics, random_state=1,
      max_iter=self.iters)

  def get_documents_by_topics(self):
    np.seterr(divide='ignore', invalid='ignore')
    docs_by_topics = self.nmf.fit_transform(self.tdm)
    normalized = docs_by_topics / np.sum(docs_by_topics, axis=1, keepdims=True)
    return np.nan_to_num(normalized) # zero out nan's

  def get_doc_to_topics(self):
    '''
    Find the distribution of each topic in each document
    '''
    doc_to_topics = defaultdict(lambda: defaultdict())
    for doc_id, topic_vals in enumerate(self.documents_by_topics):
      for topic_id, topic_presence_in_doc in enumerate(topic_vals):
        doc = self.infiles[doc_id]
        doc_to_topics[doc][topic_id] = topic_presence_in_doc
    return doc_to_topics

  def get_topic_to_words(self):
    '''
    Find the top words for each topic
    '''
    topic_to_words = defaultdict(list)
    for topic_id, topic in enumerate(self.topics_by_terms):
      top_features = topic.argsort()[:-self.words - 1:-1]
      topic_to_words[topic_id] = [self.feature_names[i] for i in top_features]
    return topic_to_words

  def write_results(self):
    self.write_json('doc_to_topics.json', self.docs_to_topics)
    self.write_json('topic_to_words.json', self.topics_to_words)

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Build topic models with Non-Negative Matrix Factorization')
  parser.add_argument('-files', required=True, help='path to directory with files to process')
  parser.add_argument('-encoding', default='utf8', help='encoding of infiles')
  parser.add_argument('-max_files', default=None, type=int, help='max files to process')
  parser.add_argument('-topics', default=20, type=int, help='number of topics to model')
  parser.add_argument('-iters', default=100, type=int, help='number of iterations to run')
  parser.add_argument('-words', default=10, type=int, help='number of words to write per topic')
  parser.add_argument('-write_output', default=True, type=bool, help='specify whether to write output JSON')
  parser.add_argument('-min_count', default=2, type=int, help='terms with >= min_count will be kept in the tdm')
  parser.add_argument('-max_freq', default=0.95, type=float, help='terms occuring in > max_freq documents will be removed')
  args = parser.parse_args()

  NMF(files=args.files, encoding=args.encoding, max_files=args.max_files,
      topics=args.topics, iters=args.iters, words=args.words, write_output=args.write_output,
      min_count=args.min_count, max_freq=args.max_freq)