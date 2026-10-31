# 🧵 FastText: A Field Guide to Word Embeddings

> **Words are made of more than whole words.**  
> FastText looks inside a word—at its character fragments—to learn representations that can capture meaning, spelling, and morphology.

## 🗺️ Trail Map

- [What Is FastText?](#-what-is-fasttext)
- [What Is a Word Embedding?](#-what-is-a-word-embedding)
- [The Key Idea: Character N-grams](#-the-key-idea-character-n-grams)
- [How FastText Learns Word Vectors](#-how-fasttext-learns-word-vectors)
- [FastText for Text Classification](#-fasttext-for-text-classification)
- [Why Subwords Matter](#-why-subwords-matter)
- [FastText, Word2Vec, and GloVe](#-fasttext-word2vec-and-glove)
- [Strengths and Limitations](#️-strengths-and-limitations)
- [Python Example: Train Word Vectors](#-python-example-train-word-vectors)
- [Python Example: Classify Text](#-python-example-classify-text)
- [Using Pretrained Vectors](#-using-pretrained-vectors)
- [Field Notes](#-field-notes)
- [Further Reading](#-further-reading)

## 🔦 What Is FastText?

**FastText** is an open-source library and method for learning word representations and performing text classification. It was developed at Facebook AI Research. Its word-vector method builds on ideas from Word2Vec, while also representing words through their character-level parts, called **character n-grams**. ([fasttext.cc](https://fasttext.cc/), [Bojanowski et al.](https://arxiv.org/abs/1607.04606))

This helps FastText learn from a word’s internal structure. For example, words such as `walk`, `walked`, and `walking` share character fragments. Their representations can therefore share information, even when the full words appear with different frequencies.

FastText is also the name of a software library that supports efficient text classification. These are related uses of the same library, but **word-vector learning** and **supervised text classification** are distinct tasks.

## 🧩 What Is a Word Embedding?

A computer needs numbers to work with words mathematically. A **word embedding** represents a word as a vector—a list of numbers:

```text
walking → [0.12, -0.37, 0.81, ...]
```

The values are learned from text. Words used in similar contexts often have vectors that are close together.

Traditional Word2Vec and GloVe assign a learned vector to each word in their vocabulary. FastText adds vectors for parts of words, allowing information to be shared across related word forms.

## 🔍 The Key Idea: Character N-grams

An **n-gram** is a sequence of `n` items. A character n-gram is a short sequence of characters taken from a word.

FastText marks a word’s boundaries and extracts character fragments from it. For illustration, a word like `where` might be represented with fragments such as:

```text
<wh, whe, her, ere, re>
```

The exact fragments depend on the configured minimum and maximum n-gram lengths and the implementation’s boundary conventions.

A word representation is built from its character n-gram representations, along with a representation for the whole word. This means related forms can share many of the same learned components.

## 🧠 How FastText Learns Word Vectors

FastText’s word-representation method extends the Skip-gram approach:

1. It reads text and identifies a target word and nearby context words.
2. It splits the target word into character n-grams.
3. It combines the target word’s n-gram vectors to represent it.
4. It uses that representation to predict nearby context words.
5. It adjusts the vectors based on prediction errors.

In simplified form:

```text
Word → Character n-grams → Combined representation → Predict context
```

The FastText research paper describes representing a word as a bag of character n-grams and using their vector representations together in a Skip-gram model. ([Bojanowski et al.](https://aclanthology.org/Q17-1010/))

The phrase **bag of n-grams** means the model uses the set of character fragments, rather than treating them as a full character sequence model that understands every spelling rule.

## 🏷️ FastText for Text Classification

FastText can also train supervised text classifiers. In this mode, it learns to assign labels to text, such as:

- `positive` or `negative` sentiment
- A topic such as `sports`, `technology`, or `business`
- A category for a support request

The library represents words and text using learned features, then trains a classifier to predict labels. FastText’s original work on text classification focused on efficient models that can be trained quickly and used with relatively modest computing resources. ([Joulin et al.](https://aclanthology.org/E17-2068/))

A classifier needs labeled examples. For instance:

```text
__label__positive I loved the film
__label__negative The service was disappointing
```

The `__label__` prefix is the convention used by the official fastText Python interface for supervised training data. ([fastText Python documentation](https://fasttext.cc/docs/en/python-module.html))

## 🌱 Why Subwords Matter

### Rare words

A rare word may not appear often enough for a traditional word-level method to learn a reliable vector. If its character fragments also occur in more common words, FastText can use information from those shared fragments.

### Word forms and morphology

In languages with many word endings, prefixes, or inflections, related forms can share character n-grams. This can help connect words such as:

```text
connect
connected
connecting
connection
```

The benefit depends on the language and corpus. Character overlap does not always mean that words have related meanings.

### Out-of-vocabulary words

A FastText model with access to its learned subword information can construct a vector for many words that were not seen as complete words during training. It does this from the character n-grams it knows.

This does **not** mean every unseen word will receive a useful representation. A misspelling, name, or unusual token may contain fragments with little relevant training information. A plain exported word-vector file may also not retain all the subword information needed to create vectors for unseen words; check the format and library documentation.

## ⚖️ FastText, Word2Vec, and GloVe

| Method | Main learning signal | Uses character subwords? | Can create vectors for many unseen word forms? |
|---|---|---:|---:|
| Word2Vec | Predicts words from context, or context from words | No | No, not in the standard model |
| GloVe | Learns from global word co-occurrence statistics | No | No, not in the standard model |
| FastText | Uses a Word2Vec-style objective with character n-grams | Yes | Often, if the subword model is available |

All three methods typically produce **static word embeddings**: a word’s vector does not change according to its sentence. FastText’s special advantage is incorporating word-internal character information, not generating context-specific meanings.

## ⚠️ Strengths and Limitations

### Strengths

- Shares information across words with common character fragments.
- Can better represent rare word forms than word-only embeddings.
- Can construct vectors for many unseen forms when subword data is available.
- Is useful for languages with rich morphology and for text containing varied word forms.
- Provides efficient supervised text-classification tools.

### Limitations

#### Character similarity is not the same as meaning

Two words can share character fragments without being semantically related. Subword information helps, but it does not guarantee a correct interpretation.

#### Static word representations

A word generally has one embedding regardless of context. For example, the word `bank` has different meanings in “river bank” and “bank account.” FastText does not automatically create a different vector for each meaning.

#### Bias in training data

FastText learns from the corpus it is given. It can reflect stereotypes, imbalances, and other patterns found in that data.

#### Preprocessing matters

Tokenization, casing, punctuation, encoding, and label formatting can affect the training results. Preprocess training and test data consistently.

#### Not a complete language model

FastText can provide word vectors and text classifiers, but it does not generate fluent, context-aware responses like a conversational language model.

## 🧪 Python Example: Train Word Vectors

The official Python interface supports unsupervised training using Skip-gram or CBOW. ([fastText Python documentation](https://fasttext.cc/docs/en/python-module.html))

Install the package:

```bash
pip install fasttext
```

Create a UTF-8 text file named `corpus.txt`, with one or more sentences per line:

```text
The cat sat by the window.
A small kitten watched the birds.
The dog ran across the green field.
```

Train a Skip-gram model:

```python
import fasttext

model = fasttext.train_unsupervised(
    "corpus.txt",
    model="skipgram",
    dim=50,
    minn=3,
    maxn=6,
    epoch=10
)

print("Vector for 'kitten':")
print(model.get_word_vector("kitten"))

print("\nWords most similar to 'kitten':")
print(model.get_nearest_neighbors("kitten", k=5))
```

This tiny corpus only demonstrates the API. Meaningful embeddings require a sufficiently large, relevant corpus.

### Important parameters

- `model`: Choose `"skipgram"` or `"cbow"`.
- `dim`: Number of dimensions in each vector.
- `minn`: Minimum character n-gram length.
- `maxn`: Maximum character n-gram length.
- `epoch`: Number of passes over the training data.

FastText’s defaults and available options can vary by interface version. Consult the documentation for the version you install.

## 🧪 Python Example: Classify Text

For supervised training, FastText expects labeled examples in a text file. Each line contains one or more labels and the text.

Create `train.txt`:

```text
__label__positive I really enjoyed this film
__label__positive The story was wonderful
__label__negative I disliked the ending
__label__negative The movie was boring
```

Create `test.txt` using the same format, with separate examples for evaluation:

```text
__label__positive The characters were excellent
__label__negative I did not enjoy the movie
```

Train and evaluate a classifier:

```python
import fasttext

model = fasttext.train_supervised(
    input="train.txt",
    epoch=25,
    lr=1.0,
    wordNgrams=2,
    verbose=0
)

examples, precision, recall = model.test("test.txt")

print(f"Test examples: {examples}")
print(f"Precision at 1: {precision:.3f}")
print(f"Recall at 1: {recall:.3f}")

labels, probabilities = model.predict("The acting was wonderful")

print("Predicted label:", labels[0])
print("Confidence:", probabilities[0])
```

A very small dataset is not enough for a reliable classifier. In a real project, use representative labeled examples, keep test examples separate from training data, and evaluate performance on data the model did not train on.

## 📦 Using Pretrained Vectors

Pretrained FastText vectors can save time when a suitable model already exists. Before choosing one, check:

- Which corpus and language it was trained on
- Whether it uses cased or lowercased text
- Its vector dimensions and vocabulary
- Whether the model format preserves subword information
- Its license and terms of use

The official fastText website provides information about models, word representations, and text classification. ([fasttext.cc](https://fasttext.cc/))

## 📏 Evaluating FastText

Evaluate FastText according to the task you intend to solve.

For word vectors, check:

- Whether nearest neighbors make sense for your domain
- Vocabulary coverage, especially for rare words
- Similarity results on suitable test examples
- Bias and unwanted associations
- Performance in a downstream task

For text classification, use held-out test data and metrics appropriate to the task, such as precision, recall, and F1-score. Accuracy alone may hide poor performance on rare or imbalanced classes.

## 📝 Field Notes

- FastText is both a library and a method for learning word vectors and classifying text.
- Its word embeddings use character n-grams as well as whole-word information.
- Shared subwords can help represent rare words and many unseen word forms.
- Subword similarity does not guarantee semantic similarity.
- FastText embeddings are static; a word’s vector usually does not change with sentence context.
- Corpus quality, preprocessing, model settings, and evaluation all matter.

## 🔗 Further Reading

- Bojanowski et al., [Enriching Word Vectors with Subword Information](https://aclanthology.org/Q17-1010/)
- Joulin et al., [Bag of Tricks for Efficient Text Classification](https://aclanthology.org/E17-2068/)
- [Official fastText Website](https://fasttext.cc/)
- [fastText Python Module Documentation](https://fasttext.cc/docs/en/python-module.html)
- [fastText GitHub Repository](https://github.com/facebookresearch/fastText)

---

*FastText looks beyond the whole word. By learning from character fragments, it can share useful information across word forms that a word-only embedding treats separately.*
