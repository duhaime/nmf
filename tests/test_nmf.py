import os, sys, glob, types, pytest, six
sys.path.append('.')
sys.path.append('..')
sys.path.append(os.path.join('..', 'nmf'))
import nmf

@pytest.fixture
def _nmf():
  return nmf.NMF(files='texts')

def test_read_returns_string(_nmf):
  f = _nmf.read_file('requirements.txt', 'utf8')
  assert isinstance(f, six.string_types)

def test_build_corpus_returns_generator(_nmf):
  files = glob.glob('*.txt')
  corpus = _nmf.build_corpus(files, 'utf8')
  assert isinstance(corpus, types.GeneratorType)

def test_get_infiles_returns_list(_nmf):
  infiles = _nmf.get_infiles('.', None)
  assert isinstance(infiles, list)

def test_tdm_row_count_equals_n_infiles(_nmf):
  assert _nmf.tdm.shape[0] == len(_nmf.infiles)

def test_docs_by_topics_rows_equals_n_infiles(_nmf):
  assert _nmf.documents_by_topics.shape[0] == len(_nmf.infiles)

def test_docs_by_topics_cols_equals_n_topics(_nmf):
  assert _nmf.documents_by_topics.shape[1] == _nmf.topics

def test_topics_by_terms_rows_equals_n_topics(_nmf):
  assert _nmf.topics_by_terms.shape[0] == _nmf.topics