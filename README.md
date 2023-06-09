# Tree-Structured Long Short-Term Memory Networks
A [PyTorch](http://pytorch.org/) based implementation of Tree-LSTM from Kai Sheng Tai's paper
[Improved Semantic Representations From Tree-Structured Long Short-Term Memory
Networks](http://arxiv.org/abs/1503.00075).

### Requirements
- [PyTorch](http://pytorch.org/) Deep learning library
- [tqdm](https://github.com/tqdm/tqdm): display progress bar
- [meowlogtool](https://pypi.python.org/pypi/meowlogtool): a logger that write everything on console to file
- Java >= 8 (for Stanford CoreNLP utilities)
- Python >= 3

## Usage
First run the script `./fetch_and_preprocess.sh`

This downloads the following data:
  - [Stanford Sentiment Treebank](http://nlp.stanford.edu/sentiment/index.html) (sentiment classification task)
  - [Glove word vectors](http://nlp.stanford.edu/projects/glove/) (Common Crawl 840B) -- **Warning:** this is a 2GB download!

and the following libraries:

  - [Stanford Parser](http://nlp.stanford.edu/software/lex-parser.shtml)
  - [Stanford POS Tagger](http://nlp.stanford.edu/software/tagger.shtml)

### Sentiment classification

```
python sentiment.py --name <name_of_log_file> --model_name <constituency|dependency> --epochs 10
```
