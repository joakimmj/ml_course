# Course in machine learning (ML)
A simple introduction to text classification with machine learning. The course will cover some elementary linguistic morphology techniques for extracting features, and an introduction to some classifiers.

The tasks in this course will be solved in **Python 3** with the **scikit-learn** tool. 

__Tasks__:
1. Creating a spam-filter
2. Sentiment analysis of movie reviews

> **Prerequisites**: Some *Python*

### Document classification/categorization
The process is to assign a document to one or more classes or categories. The documents may be texts, images, music, etc. One of the most common types of document classification is text classification. 

* **Content-based** classification is classification in which the weight given to particular subjects in a document determines the class to which the document is assigned. It could be the number of times given words appears in a document.
* **Request-oriented** classification (or indexing) is classification in which the anticipated request from users is influencing how documents are being classified.

### Linguistic morphology 
Before we can use the data set, we have to extract the features we want to use. There are several ways to extract these features. Some normal techniques are:

1. **Tokenization** is the process of delimiting a string of input characters. The resulting tokens are then passed on to some other form of processing.  
E.g. `The quick brown fox jumps over the lazy dog => [the, quick, brown, fox, jumps, over, the, lazy, dog]`
2. **Stemming** is the process of reducing inflected (or sometimes derived) words to their word stem, base or root form.  
E.g. `cats, catlike, catty => cat`, or the special form where the stem is not itself a word `argue, argued, argues, arguing => argu`
3. **Lemmatization** is the process of grouping together the inflected forms of a word so they can be analysed as a single item, identified by the word's lemma, or dictionary form.  
E.g. `am, are, is => be`
4. Removing **stop words** is a way to remove irrelevant data. Any group of words can be chosen as the stop words for a given purpose. This means that stop words are the words that don't provide any context for the given task at hand.  
E.g. `a, and, as, at, in, the`
5. The **bag-of-words** (BoW) model is used to look at the frequency of each word in a document. All words in the corpus (text samples) form a dictionary.  
E.g.: The dictionary `["John", "likes", "to", "watch", "movies", "Mary", "too", "also", "football", "games"]` 
applied to the text `John likes to watch movies. Mary likes movies too.` will form the vector `[1, 2, 1, 1, 2, 1, 1, 0, 0, 0]`. 

### Choosing the right estimator
We want to make a model that predicts a class for different text examples. For simplicity we have chosen data sets with binary classification (e.g. spam/ham). 

For choosing the right estimator we can use scikit's estimator chooser:
![Estimator picker](files/ml_map.png)

## Setup
setup instructions
Install python 3

### Windows
install anaconda

### Linux/MacOS
```bash
pip install -r requirements.txt
```

## Tasks
print wrongly classified ??

### Task 1
spam filter
> **Tip:** Use some of the linguistic morphology techniques mentioned above.

### Task 2
sentiment analysis of movie reviews

----------

