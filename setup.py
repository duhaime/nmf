from setuptools import setup

setup (
  name='nmf',
  version='0.0.5',
  packages=['nmf'],
  keywords = ['data-mining', 'text-mining', 'topic-modeling', 'nmf'],
  description='Non-negative matrix factorization for building topic models in Python',
  url='https://github.com/duhaime/nmf',
  author='Douglas Duhaime',
  author_email='douglas.duhaime@gmail.com',
  license='MIT',
  install_requires=[
    'numpy==1.12.1',
    'scikit-learn==0.18.1'
  ],
)