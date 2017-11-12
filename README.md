# NMF Topic Models

[![Build Status](https://travis-ci.org/duhaime/nmf.svg?branch=master)](https://travis-ci.org/duhaime/nmf)

[Non-Negative Matrix Factorization](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization) is a dimension reduction technique that factors an input matrix of shape *m x n* into a matrix of shape *m x k* and another matrix of shape *n x k*.

In text mining, one can use NMF to build [topic models](https://en.wikipedia.org/wiki/Topic_model). Using NMF, one can factor a [Term-Document Matrix](https://en.wikipedia.org/wiki/Document-term_matrix) of shape *documents x word types* into a matrix of *documents x topics* and another matrix of shape *word types x topics*. The former matrix describes the distribution of each topic in each document, and the latter describes the distribution of each word in each topic.

Given a collection of input documents, the source code in this repository builds a memory-efficient Term-Document Matrix, factors that matrix using NMF, then writes the resulting data structures as [JSON outputs](#ouput-data-json).

## Usage

#### Command Line Usage

```
# Obtain sample documents
wget https://s3.amazonaws.com/duhaime/github/nmf/texts.tar.gz
tar -zxf texts.tar.gz && rm texts.tar.gz

# Install dependencies
pip install -r requirements.txt --user

# Build a topic model with 20 topics using ./texts/ as the input directory
python nmf.py -files texts -topics 20
```

For a list of all command line options, run `python nmf.py -h`

#### Class Usage

```
import nmf
model = nmf.NMF(files='texts', topics=20, write_output=False)

# access the documents by topics matrix
model.documents_by_topics

# access the topics by terms matrix
model.topics_by_terms
```

### JSON Output

**topic_to_words.json** maps each topic id to the top words in that topic:

```
{
  "0": [
    "colours",
    "light",
    "prism", ...
  ],
  "1": [
    "sap",
    "tree",
    "bark", ...
  ], ...
}
```

**doc_to_topics.json** maps each input document to each topic id and its weight in the document:

```
{
  "texts/doc_1.txt": {
    "0": 0.52,
    "1": 0.0,
    "2": 0.0, ...
  },
  "texts/doc_2.txt": {
    "0": 0.0,
    "1": 0.67,
    "2": 0.0, ...
  },
]
```