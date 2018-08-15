---
layout: post
title: "To embed or not to embed: the question for sentiment analysis"
authors: Ilya & Vanya
---
# Intro

Word embeddings are very popular for NLP tasks. Some data scientists use pretrained "out-of-the-box" word embeddings to classify sentiment. However, it was [pointed out](https://github.com/attardi/deepnl/wiki/Sentiment-Specific-Word-Embeddings) that generic word embeddings produce vector space representations that do not necessary well encode sentiment information by proximity relationships. To cite the *deepnl* Python package documentation:

> Word embeddings are typically learned from unannotated plain text and provide a dense vector representation of syntactic/semantic aspects of a word. These representations though are not able to distinguish contrasting aspects of a word sense, for example sentiment polarity or opposite senses (e.g. high/low).


In this post, we are going to verify this claim and compare several pretrained embeddings in a sentiment analysis task.

# Our setup

* Data: Opinion lexicon dataset  by Liu and Hu <https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon>. 
  
  The dataset contains ~7000 English words with positive and negative sentiment (binary labels)
  
  
* Several pre-trained word embeddings: Google's Word2Vec, Glove, MUSE.

## Choice of appropriate text data representation method

What is the most efficient method to numerically represent text data to carry out text classification?

### Classical approach: docs as vectors

In text classification, a standard data preprocessing step is to represent text documents as numerical vectors. For example, the almighty bag-of-words (BOW) model represents a text document with $(c_1 \ c_2\ ...\ c_n)^\text{T} \in \mathbb{R}^n$ where $c_i$ is the number of occurences of the word $i$, $n$ is the size of the vocabulary. A slightly more complicated version of BOW is to replace raw word count with TF-IDF (Term Frequency times Inverse Document Frequency) whereby the frequency of a word occurence in a document is normalized by its frequency across documents.

### Zoo of embeddings: words as vectors

Instead of representing whole documents as vectors one can represent single words. A simplest way to do it is one-hot encoding vector for vocabulary of size $n$: a word $j \in \\{1...n\\}$ is represented by $(b_1\ b_2\ ...\ b_n)^\text{T}$ where all $b_i = \delta_{ij}$.
After one has a vector space representation, one can learn mapping from the vector space with one dimension per word to a lower dimensional continuous vector space. To learn this embedding one can construct objective functions so that the learned mapping reflects syntactic relationshits between words. For this purpuse, neural network based embeddings become increasingly popular: [word2vec](https://en.wikipedia.org/wiki/Word2vec) by Google, [MUSE](https://research.fb.com/downloads/muse-multilingual-unsupervised-and-supervised-embeddings/) by facebook, [GloVe](https://en.wikipedia.org/wiki/GloVe) at Stanford, FastText. The advantage of these more modern approaches is that they can learn syntactic relationships that allow for meaningful word-vector distances.

# What's up with embeddings?

In this post we wish to see how word sentiment is reflected in the structure of embedding word-vector spaces of popular embedding models. For this purpose we will use sentiment-specific words of the English language and compare how well words with opposite polarity are separated in embedding space for pretrained GloVe, Word2Vec and MUSE.

But first, some imports:


```python
import os
from copy import deepcopy

import pandas as pd
import numpy as np
import scipy.spatial.distance as distance
from statistics import mode

# to compute KS distance between distributions:
from scipy.stats import ks_2samp 

import matplotlib.pylab as plt
from matplotlib.lines import Line2D
from pandas.plotting import radviz

import gensim

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.decomposition import PCA

try:
    # https://github.com/DmitryUlyanov/Multicore-TSNE:
    from MulticoreTSNE import MulticoreTSNE as TSNE
except ImportError:
    print('Using sklearn\'s t-SNE implementation')
    from sklearn.manifold import TSNE
    
from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.optimizers import Adam
```


```python
# nice plots in jupyter notebooks
import pylab
import seaborn as sns

sns.set(font_scale=1.5)
pylab.rcParams['figure.figsize'] = 13, 8
```

## Downloading embeddings (can place under a cut in a post)

Pre-trained MUSE and GloVe in plain-text format are accessible for direct download from the web via *wget* or *curl*.

A binary file of the word2vec model trained on the Google news corpus is hosted on Google Drive: <https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit>

Below is a simple Python routine to download the file from Google Drive (working as of August 2018):

<details> 
  <summary>
      Google downloader
  </summary>


```python
import requests

def download_file_from_google_drive(id, destination):
    
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, 'wb') as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = 'https://docs.google.com/uc?export=download'

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    
```

</details>


```python
FILE_ID = '0B7XkCwpI5KDYNlNUTTlSS21pQmM'

download_file_from_google_drive(FILE_ID,
                                './data/GoogleNews-vectors-negative300.bin.gz')
```

## Loading word embeddings

We load MUSE embeddings that output a 300-dimensional real-valued vector, taking a token as an input. The pre-trained MUSE model can be downloaded from <https://github.com/facebookresearch/MUSE>, and is essentially a plain-text representation of the word-vector dictionary.
We also loaded GloVe and word2vec trained on Google news (download from <http://nlp.stanford.edu/data/glove.twitter.27B.zip>, <https://code.google.com/archive/p/word2vec/>): 


```python
def load_embedding(path, 
                   full_vocab=False):
    '''Load a word-vector dictionary from plain text'''
    
    embedding_dict = dict()
    
    with open(path, 'r',
              encoding='utf-8', newline='\n',
              errors='ignore') as datafile:
        
        for line in datafile:
            word, vector = line.rstrip().split(' ', 1)
            if not full_vocab:
                word = word.lower()
            vector = np.fromstring(vector, sep=' ')
            embedding_dict[word] = vector
    
    mode_len = mode([len(vector) for vector in embedding_dict.values()])
    embedding_dict = {word: vector for word, vector 
                      in embedding_dict.items() if len(vector) == mode_len}
    
    return embedding_dict
```


```python
%%time

embeddings = {}

# loading MUSE
embeddings['muse'] = load_embedding('./data/muse_trained/wiki.multi.en.vec')

# loading Word2Vec
w2v_dim = 300
embeddings['word2vec'] = gensim.models.KeyedVectors.load_word2vec_format(
                            './data/w2v_trained/GoogleNews-vectors-negative300.bin', binary=True)
embeddings['word2vec'] = dict(zip(embeddings['word2vec'].vocab.keys(), embeddings['word2vec'].vectors))
embeddings['word2vec'] = {word: vector for word, vector in embeddings['word2vec'].items() 
                          if len(vector) == w2v_dim and word[0].islower()}

# loading GloVe
embeddings['glove'] = load_embedding('./data/glove_trained/glove.twitter.27B.200d.txt')
```

    CPU times: user 2min 12s, sys: 4.92 s, total: 2min 17s
    Wall time: 2min 18s



```python
# let's see the dimensionality of our embedding word-vectors:

test_word = 'perestroika'

embeddings['muse'][test_word].shape, \
embeddings['word2vec'][test_word].shape, \
embeddings['glove'][test_word].shape
```




    ((300,), (300,), (200,))



## Separation of labeled sentiment-specific words in the embedding space


We will use a dataset of sentiment-specific words together with embeddings to see how well these words are separated inthe embedding space.

Let's use the [Liu and Hu opinion lexicon](https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon) dataset for our tests. The dataset contains around 6800 positive and negative opinion words or sentiment words for the English language. The dataset is not large, but it will give us an intuition of how sentiment-specific words are structured in embedding spaces.


[direct download link](https://www.cs.uic.edu/~liub/FBS/opinion-lexicon-English.rar)



```python
def load_words_liu(folder):
    
    def file_to_list(filepath):
        header_lines = 35
        with open(filepath, 'r', encoding='latin-1') as file:
            return [line.strip() for index, line in enumerate(file) 
                    if index >= header_lines]
    
    list_positive = file_to_list(os.path.join(folder, 
                                              'positive-words.txt'))
    list_negative = file_to_list(os.path.join(folder, 
                                              'negative-words.txt'))
    
    return {'pos': set(list_positive), 
            'neg': set(list_negative)}

words = load_words_liu('./data/opinion_lexicon/')
```

Let's limit the amount of both positive and negative sentiment words in our data:


```python
words_num = 2000     
# a part of these words would not be encoded by embedding models
# simply due to absense of the word in the initial corpus
    
words = {sentiment: set(list(word_list)[:words_num]) 
         for sentiment, word_list in words.items()}
```

Let's define a function that embeds all words from a list of words


```python
def embed_from_list(embedding, word_list):
    
    list_embedded = [embedding[word] for word in word_list
                     if word in embedding]
    
    if len(list_embedded) > 0:
        return np.vstack(list_embedded)
    else:
        return None
```

We now embed all words corresponding to positive and negaive classes with all embedding models.

If we plot the pairwise distance distributions for the two classes of words, we will see that althouth distributions largely overlap, separation between the negative and positive word-vector clouds exists and differs across embeddings: better separation can be seen for Word2Vec and MUSE than for GloVe (a fact also confirmed by [Kolmogorov-Smirnov](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test) statistics).


```python
for name, embedding in embeddings.items():
    
    # Pairwise distance histograms:
    plt.figure()
    pairwise_dist_distrib = {}
    
    for sentiment in ['pos', 'neg']:
        
        pairwise_dist_distrib[sentiment] = \
            distance.pdist(embed_from_list(embedding, words[sentiment])).flatten()
        plt.hist(pairwise_dist_distrib[sentiment],
                 label=sentiment, density=True, alpha=0.4, bins=50)
        plt.xlabel('Inter-vector distance')
    
    pairwise_dist_distrib['pos-neg'] = \
        distance.cdist(embed_from_list(embedding, words['pos']),
                       embed_from_list(embedding, words['neg'])).flatten()
    plt.hist(pairwise_dist_distrib['pos-neg'],
             label='pos-neg', density=True, alpha=0.4, bins=50)
    plt.title(name + ' embedding')
    plt.xlabel('Inter-vector distance')
    plt.legend()
    
    # KS statistics:
    print(name)
    print(pd.DataFrame.from_records(
        [ [k1, k2, ks_2samp(pairwise_dist_distrib[k1],
                            pairwise_dist_distrib[k2])[0]]
          for k1 in pairwise_dist_distrib
             for k2 in pairwise_dist_distrib
                 if k1 < k2
        ], columns=['first', 'second', 'KS-distance']))
    print('\n')
    
```

    muse
      first   second  KS-distance
    0   pos  pos-neg     0.108299
    1   neg      pos     0.060788
    2   neg  pos-neg     0.167416
    
    
    word2vec
      first   second  KS-distance
    0   pos  pos-neg     0.170442
    1   neg      pos     0.179398
    2   neg  pos-neg     0.028857
    
    
    glove
      first   second  KS-distance
    0   pos  pos-neg     0.045027
    1   neg      pos     0.019457
    2   neg  pos-neg     0.059782
    
    



![png](/images/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_files/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_21_1.png)



![png](/images/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_files/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_21_2.png)



![png](/images/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_files/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_21_3.png)


Let's define a utility function that converts words of different sentiment to a design matrix and a label vector in a standard *sklearn* format:


```python
def vectorize(embedding, words):
    
    data_matrix, labels = [], []
    for sentiment in ['pos', 'neg']:
        X_ = embed_from_list(embedding, 
                             words[sentiment])
        data_matrix.append(X_)
        labels.append(np.array([sentiment] * X_.shape[0]))
    
    return np.vstack(data_matrix), np.hstack(labels)
```

Let's define a function that plots 2-dimensional scatter plots of word-vectors after a dimensionality reduction technique is applied to the data (e.g. 2-component PCA transform):


```python
def plot_lowdim_embed(dim_reducer_, embeddings, words,
                      alpha=0.2,
                      subsample=1000 # subsample length for better viz
                      ):
    
    fitted_dimreducers = {}
    fig, axs = plt.subplots(1, len(embeddings),
                            figsize=(6 * len(embeddings), 5))
    if subsample is not None:
        words = {sentiment: set(list(word_list)[:subsample]) 
                 for sentiment, word_list in words.items()}
    
    for ax, (name, embedding) in zip(axs, embeddings.items()):
        X, y = vectorize(embedding, words)
        y_ = pd.Series(y).map({'neg': 0, 
                               'pos': 1})
        try:
            dim_reducer = deepcopy(dim_reducer_)
        except:
            dim_reducer = dim_reducer_

        lowdim_embed = dim_reducer.fit_transform(X, y_)
        fitted_dimreducers[name] = dim_reducer
        ax.scatter(lowdim_embed[:, 0],
                   lowdim_embed[:, 1], 
                    color=pd.Series(y).map(
                        {'pos': 'red',
                         'neg': 'blue'}),
                    alpha=alpha)
        ax.set_title(name)
    
    return fitted_dimreducers
```

We can apply PCA straightaway (embedded vectors are normalized) and by taking the first 2 principal components we get these plots:


```python
print('Red points represent positive sentiment, ' \
      'blue points - negative sentiment')
plot_lowdim_embed(PCA(n_components=2), embeddings, words);
```

    Red points represent positive sentiment, blue points - negative sentiment



![png](/images/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_files/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_27_1.png)


We can also apply a nonlinear mapping such as 2D t-SNE to get a nicer visual separation of sentiment classes:


```python
%%time 
print('Red points represent positive sentiment, ' \
      'blue points - negative sentiment')

_ = plot_lowdim_embed(TSNE(n_jobs=8,
                           random_state=1), 
                      embeddings, words)
```

    Red points represent positive sentiment, blue points - negative sentiment
    CPU times: user 1min 1s, sys: 364 ms, total: 1min 1s
    Wall time: 1min 1s



![png](/images/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_files/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_29_1.png)


Let's visualize the first 6 principal components of the data with [RadViz](https://pandas.pydata.org/pandas-docs/stable/visualization.html#visualization-radviz):


```python
fig, axs = plt.subplots(1, len(embeddings),
                        figsize=(6 * len(embeddings), 5))

word_num_cutoff = 500
words_ = {sentiment: set(list(word_list)[:word_num_cutoff]) 
          for sentiment, word_list in words.items()}

for ax, (name, embedding) in zip(axs, embeddings.items()):
    
    list_df = []
    X, y = vectorize(embedding, words_)
    lowdim_embed = PCA(n_components=6).fit_transform(X)
    df = pd.DataFrame(lowdim_embed)
    df['class'] = y
    radviz(df, 'class', color=['r', 'b'], alpha=0.08, ax=ax)
    ax.set_title(name)
```


![png](/images/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_files/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_31_0.png)


From the above plots we see that there's clearly some separation between word-vector high density parts of clouds corresponding to different sentiment polarity. However for a given region in word-vector space where a certain sentiment is dominant, there still exist points belonging to an opposite sentiment class.

Some of the embedding space dimensions are probably more useful in encoding sentiment than others.
One way to detect such dimensions is to evaluate feature importance scores of embedding components in a sentiment classification task.
Here we use logistic regression with an L1 regularization term to classify sentiment of individual words for this purpose (tree-based models are also very useful for this kind of task):


```python
logit = LogisticRegression(C=1.,
                           penalty='l1',
                           random_state=42)

coefficients = {}

for name, embedding in embeddings.items():
    
    X, y = vectorize(embedding, words)
    
    data_split = train_test_split(X, y, 
                                  test_size=0.5)
    X_train, X_test, y_train, y_test = data_split

    logit.fit(X_train, y_train)
    print(name)
    print(round(accuracy_score(y_test, 
                               logit.predict(X_test)), 4), '\n')
    
    coefficients[name] = logit.coef_.flatten()
```

    muse
    0.9256 
    
    word2vec
    0.9353 
    
    glove
    0.8613 
    


We've got good accuracy here, which means that it is possible to separate sentiment specific words with a hyperplane in the embedding space for all three embedding models.

We now wish to check which embedding dimensions were useful in sentiment classification by examining absolute values of logit coefficients:


```python
print('Logit with L1 reg coefficient values for different embeddings:\n')

large_coefs = {}

fig, axs = plt.subplots(1, len(coefficients), 
                        figsize=(5 * len(coefficients), 5))

for ax, (name, coefs) in zip(axs, coefficients.items()):
    ax.hist(coefs, bins=50)
    ax.set_title(name)
    large_coefs[name] = coefs > np.median(coefs)
    print(name, '\n# of large coefficients ', 
          len(coefs[large_coefs[name]]), '\n')
    
fig.tight_layout()
```

    Logit with L1 reg coefficient values for different embeddings:
    
    muse 
    # of large coefficients  50 
    
    word2vec 
    # of large coefficients  74 
    
    glove 
    # of large coefficients  89 
    



![png](/images/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_files/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_37_1.png)


We can now cosider reduced embedding spaces which are only comprised of embedding dimensions relevant for sentiment classification:


```python
short_embeddings = {}

for embedding in embeddings:
    short_embeddings[embedding] = {word: vector[large_coefs[embedding]]
                                   for word, vector in embeddings[embedding].items()}
```


```python
test_word = 'perestroika'

short_embeddings['muse'][test_word].shape, \
short_embeddings['word2vec'][test_word].shape, \
short_embeddings['glove'][test_word].shape, \
```




    ((50,), (74,), (89,))




```python
plot_lowdim_embed(PCA(n_components=2), 
                  short_embeddings, words);
```


![png](/images/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_files/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_41_0.png)



```python
_ = plot_lowdim_embed(TSNE(n_jobs=8,
                           random_state=1), 
                      short_embeddings, words)
```


![png](/images/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_files/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_42_0.png)


Compared to the PCA plots for full embedding spaces, there is visually a significant increase in separability, but there is still a considerable overlap between sentiment classes.

We will now try to learn a nonlinear mapping of word vectors to lower-dimensional space such that it maximizes separability of sentiment classes.
We will use a neural network model to learn this mapping.


```python
def model_generator(input_dim,
                    embed_dim=2,
                    optimizer = Adam(lr=0.0002, beta_1=0.5)):
    
    inp = Input(shape=(input_dim, ), name='x_input')
    result = Dropout(0.5)(inp)
    result = Dense(embed_dim, activation='tanh')(result)    
    embedder = Model(name='mapper', inputs=inp, outputs=result)
    result = Dropout(0.5)(result)
    result = Dense(1, activation='sigmoid')(result)
    classifier = Model(name='classifier', inputs=inp, outputs=result)
    classifier.compile(loss='binary_crossentropy', 
                       optimizer=optimizer, 
                       metrics=['accuracy'])
    
    return embedder, classifier
```

Next we define new dimensionality reduction class based on the neural network model constructed above:


```python
class NeuralWordMapping():
    
    def __init__(self,
                 n_components=2,
                 model_maker=model_generator,
                 verbose_visual=1):
        
        self.n_components=n_components
        self.model_maker = model_maker
        self.verbose_visual = verbose_visual
        
    def fit_transform(self, X, y):
        
        embedder, classifier = self.model_maker(X.shape[1], 
                                                embed_dim=self.n_components)
        
        h = classifier.fit(X, y, epochs=100, 
                           validation_split=0.2, verbose=0)
        
        if self.verbose_visual:
        
            plt.figure(figsize=(7, 3))
            plt.title('Train and validation accuracy vs. training epoch')
            for label in ['acc', 'val_acc']:
                plt.plot(h.history[label], label=label)
            plt.legend()
        
        self.classifier = classifier
        self.embedder = embedder
        
        return embedder.predict(X)
```


```python
%%time

_ = plot_lowdim_embed(NeuralWordMapping(n_components=2, 
                                        verbose_visual=1),
                      embeddings, words)
```

    CPU times: user 17.8 s, sys: 432 ms, total: 18.2 s
    Wall time: 18.1 s



![png](/images/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_files/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_48_1.png)



![png](/images/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_files/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_48_2.png)



![png](/images/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_files/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_48_3.png)



![png](/images/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_files/2018-08-03-to-embed-or-not-the-question-for-sentiment-analysis_48_4.png)


Now the sentiment classes look quite nicely separated compared to the PCA plots, and we were able to achieve more than 80% accuracy in sentiment classification on this dataset for MUSE and word2vec embeddings. Results were slightly worse for GloVe embeddings.

These tests demonstrate that a claim that word embedding spaces carry no information (structure) about word sentiment/polarity is not exactly right, and this structure can be nicely revealed with appropriate dimensionality reduction techniques.
