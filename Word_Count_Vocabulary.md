# Common setup (run once)

```bash
pip install nltk
```

```python
import nltk

# Download the NLTK datasets/models needed by the lab programs.
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('brown')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('universal_tagset')
nltk.download('conll2000')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
```

Run each program in Jupyter Notebook, Google Colab, or VS Code with Python installed. Some programs download corpora on first run and need an internet connection.

---

# 1. Word counting, vocabulary operations, and list processing

## (i) Count the total number of words and distinct words

```python
# Store a sample text document in a string.
text = """
She sells sea shells by the sea shore.
She sells beautiful shells.
"""

# Convert the complete text to lowercase so that words like
# "She" and "she" are treated as the same word.
text = text.lower()

# Split the text into individual words.
words = text.split()

# Count the total number of words, including repetitions.
total_words = len(words)

# Convert the word list into a set to remove duplicates.
distinct_words = set(words)

# Print the results.
print("Total number of words:", total_words)
print("Number of distinct words:", len(distinct_words))
print("Distinct words:", sorted(distinct_words))
```

### Sample output

```text
Total number of words: 12
Number of distinct words: 8
Distinct words: ['beautiful', 'by', 'sea', 'sells', 'shells.', 'she', 'shore.', 'the']
```

**Explanation:** `split()` separates the text into words, `len()` counts them, and `set()` removes duplicate entries. This basic example retains punctuation attached to words.

## (ii) Compute the vocabulary of sentences sent1 to sent8

```python
from nltk.book import sent1, sent2, sent3, sent4
from nltk.book import sent5, sent6, sent7, sent8

# Combine all eight sentences using list addition.
sentences = sent1 + sent2 + sent3 + sent4 + sent5 + sent6 + sent7 + sent8

# Convert the combined list into a set to get unique tokens.
vocabulary = set(sentences)

# Sort the vocabulary alphabetically.
sorted_vocabulary = sorted(vocabulary)

print("Total tokens:", len(sentences))
print("Vocabulary size:", len(vocabulary))
print("First 20 vocabulary items:", sorted_vocabulary[:20])
```

### Sample output

```text
Total tokens: [depends on the NLTK book sentences]
Vocabulary size: [depends on the NLTK book sentences]
First 20 vocabulary items: [alphabetically sorted words]
```

The exact numbers are obtained when the program runs with the NLTK book corpus. `sent1` through `sent8` are example sentences defined in NLTK's `book` module.

## (iii) Perform operations on the given list

```python
# Define the given list of words.
sent = ['she', 'sells', 'sea', 'shells', 'by',
        'the', 'sea', 'shore']

# (a) Print words beginning with "sh".
print("Words beginning with 'sh':")

for word in sent:
    if word.startswith("sh"):
        print(word)

# (b) Print words longer than four characters.
print("Words longer than four characters:")

for word in sent:
    if len(word) > 4:
        print(word)
```

### Sample output

```text
Words beginning with 'sh':
she
shells
shore

Words longer than four characters:
sells
shells
shore
```

**Explanation:**
- `startswith("sh")` checks the beginning of each word.
- `len(word) > 4` selects words containing more than four characters.

---
