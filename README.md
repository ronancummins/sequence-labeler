Sequence labeler
=========================

This is a multitask neural network sequence labeling system. Given a sequence of tokens, it will learn to assign labels to each token, but will also output a score at the end of each sequence. This is a modification of the original sequence labelling system such that it performs automated essay scoring while also taking into account grammatical error detection. 
As before the main model implements a bidirectional LSTM for sequence tagging. The automated essay scoring part sums the hidden states of the bidirectional LSTM (in both directions) and the uses the concatenated vectors to feed into a output layer. 
As before Run with:

    python sequence_labeling_experiment.py config.conf

Preferably with Theano set up to use CUDA, so the process can run on a GPU. The script will train the model on the training data, test it on the test data, and print various evaluation metrics.

Requirements
-------------------------

* python (tested with 2.7.6)
* numpy (tested with 1.12.0)
* theano (tested with 0.8.2)
* lasagne (tested with 0.1)

The code should also be compatible with python3.

At the time of writing, the latest released version of lasagne is not compatible with the latest released version of theano. Install the development version of lasagne to get around this.

The latest cuDNN doesn't seem to behave well with the CRF implementation. If you get weird errors with CRF activated, try disabling cuDNN with dnn.enabled=False.

Data format 
-------------------------

The training and test data is slighly modified such that a score is assigned to an id of the sequence. There are some example files in the data/ folder which shows the correct format.

For multitask automatic essay scoring and error detection, this would be something like:

    ID-1    7
    I       c
    saws    i
    the     c
    show    c

    ID-2    10
    I       c
    went    c 
    to      c
    see     c
    a       c
    movie   c
    .       c
    It      c
    was     c	

Configuration
-------------------------

Edit the values in config.conf as needed:

* **path_train** - Path to the training data, in CoNLL tab-separated format. One word per line, first column is the word, last column is the label. Empty lines between sentences.
* **path_dev** - Path to the development data, used for choosing the best epoch.
* **path_test** - Path to the test file. Can contain multiple files, colon separated.
* **main_label** - The output label for which precision/recall/F-measure are calculated.
* **conll_eval** - Whether the standard CoNLL NER evaluation should be run.
* **lowercase_words** - Whether words should be lowercased when mapping to word embeddings.
* **lowercase_chars** - Whether characters should be lowercased when mapping to character embeddings.
* **replace_digits** - Whether all digits should be replaced by 0.
* **min_word_freq** - Minimal frequency of words to be included in the vocabulary. Others will be considered OOV.
* **use_singletons** - Option for randomly mapping words with count 1 to OOVs.
* **allowed_word_length** - Maximum allowed word length, clipping the rest. Can be necessary if the text contains unreasonably long tokens, eg URLs.
* **preload_vectors** - Path to the pretrained word embeddings, in word2vec plain text format. If your embeddings are in binary, you can use [convertvec](https://github.com/marekrei/convertvec) to convert them to plain text.
* **word_embedding_size** - Size of the word embeddings used in the model.
* **char_embedding_size** - Size of the character embeddings.
* **word_recurrent_size** - Size of the word-level LSTM hidden layers.
* **char_recurrent_size** - Size of the char-level LSTM hidden layers.
* **narrow_layer_size** - Size of the extra hidden layer on top of the bi-LSTM.
* **crf_on_top** - If True, use a CRF as the output layer. If False, use softmax instead.
* **char_integration_method** - How character information is integrated. Options are: "none" (not integrated), "input" (concatenated), "attention" (the method proposed in Rei et al. (2016)).
* **dropout_input** - The probability for applying dropout. 0.0 means no dropout.
* **lmcost_gamma** - Weight for the language modeling loss. 
* **lmcost_layer_size** = Hidden layer size for the language modeling loss.
* **lmcost_max_vocab_size** = Maximum vocabulary size for the language modeling loss. The remaining words are mapped to a single entry.
* **aescost_gamma** = Weight for the automatic essay scoring loss
* **epochs** - Maximum number of epochs to run.
* **best_model_selector** - What is measured on the dev set for model selection: "dev_conll_f:high" for NER and chunking, "dev_acc:high" for POS-tagging, "dev_f05:high" for error detection.
* **stop_if_no_improvement_for_epochs** - Training will be stopped if there has been no improvement for n epochs.
* **learningrate** - Learning rate.
* **opt_strategy** - Optimisation method: sgd/adadelta/adam.
* **max_batch_size** - Maximum batch size.
* **save** - Path to save the model.
* **load** - Path to load the model.
* **garbage_collection** - Whether garbage collection is explicitly called. Makes things slower but can operate with bigger models.
* **random_seed** - Random seed for initialisation and data shuffling. This can affect results, so for robust conclusions I recommend running multiple experiments with different seeds and averaging the metrics.

The config files for the multi-task automated essay scoring task are in the results directory (called baseline.aes.conf, multitask.aes.conf, multitask.aes.lmcost.conf respectively). These are the config files that generated the results for Table 3 in the paper **Neural Multi-task Learning in Automated Assessment**.

Printing output
-------------------------

There is now a separate script for loading a saved model and using it to print output for a given input file. Use the **save** option in the config file for saving the model. The input file needs to be in the same format as the training data (one word per line, labels in a separate column). The labels are expected for printing output as well. If you don't know the correct labels, just print any valid label in that field.

To print the output, run:

    python print_output.py labels model_file input_file

This will print the input file to standard output, with an extra column at the end that shows the prediction. 

You can also use:

    python print_output.py probs model_file input_file

This will print the individual probabilities for each of the possible labels.



References
-------------------------

If you use the multi-task sequence labeling code, please reference:
[**Neural Multi-task Learning in Automated Assessment**](https://arxiv.org/pdf/1801.06830.pdf)  
Ronan Cummins and Marek Rei

If you use the main sequence labeling code, please reference:

[**Compositional Sequence Labeling Models for Error Detection in Learner Writing**](http://aclweb.org/anthology/P/P16/P16-1112.pdf)  
Marek Rei and Helen Yannakoudakis  
*In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL-2016)*
  

If you use the character-level attention component, please reference:

[**Attending to characters in neural sequence labeling models**](https://aclweb.org/anthology/C/C16/C16-1030.pdf)  
Marek Rei, Gamal K.O. Crichton and Sampo Pyysalo  
*In Proceedings of the 26th International Conference on Computational Linguistics (COLING-2016)*

If you use the language modeling objective, please reference:

[**Semi-supervised Multitask Learning for Sequence Labeling**](https://arxiv.org/abs/1704.07156)  
Marek Rei  
*In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (ACL-2017)*

The CRF implementation is based on:

[**Neural Architectures for Named Entity Recognition**](https://arxiv.org/abs/1603.01360)  
Guillaume Lample, Miguel Ballesteros, Sandeep Subramanian, Kazuya Kawakami and Chris Dyer  
*In Proceedings of NAACL-HLT 2016*
  

The conlleval.py script is from: https://github.com/spyysalo/conlleval.py


License
---------------------------

MIT License

Copyright (c) 2017 Marek Rei

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
