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

# 6. Implementation of stemming and lemmatization

**Aim:** To reduce words to their stems and dictionary base forms.

```python
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download WordNet, which the lemmatizer uses.
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize the stemmer and lemmatizer.
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Words for demonstrating stemming and lemmatization.
words = ['playing', 'played', 'studies', 'cars', 'running']

print(f"{'Word':<12}{'Stem':<12}{'Lemma':<12}")
print("-" * 36)

for word in words:
    # Stemming removes word endings using heuristic rules.
    stem = stemmer.stem(word)

    # Lemmatization finds a dictionary base form.
    # The default POS is noun, so specify verb for these examples.
    lemma = lemmatizer.lemmatize(word, pos='v')

    print(f"{word:<12}{stem:<12}{lemma:<12}")
```

### Sample output

```text
Word        Stem        Lemma
------------------------------------
playing     play        play
played      play        play
studies     studi       study
cars        car         cars
running     run         run
```

**Explanation:**

- **Stemming:** Removes or modifies word endings without necessarily producing a valid dictionary word. For example, `studies → studi`.
- **Lemmatization:** Uses vocabulary and morphological rules to find a base form. For example, `studies → study` when treated as a verb.
- Lemmatization depends on the correct part-of-speech (POS) input. Here, `cars` remains `cars` because it is interpreted as a verb by the chosen call, rather than as a plural noun.

---
