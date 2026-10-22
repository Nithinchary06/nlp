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

# 4. Demonstrate the morphology of a word using an Add-Delete table

**Aim:** To illustrate how suffixes and prefixes can be added to or deleted from a word to derive related forms.

```python
# Define the word transformations in an Add-Delete table.
# Format: original word, addition, deletion, resulting word.
morphology_table = [
    ('play', 's', '-', 'plays'),
    ('play', 'ed', '-', 'played'),
    ('play', 'ing', '-', 'playing'),
    ('playing', '-', 'ing', 'play'),
    ('played', '-', 'ed', 'play'),
    ('happy', 'ness', 'y', 'happiness'),
    ('unhappy', '-', 'un', 'happy')
]

# Print the table header.
print(f"{'Original':<12}{'Add':<10}{'Delete':<10}{'Result':<15}")
print("-" * 47)

# Display each morphological operation.
for original, add, delete, result in morphology_table:
    print(f"{original:<12}{add:<10}{delete:<10}{result:<15}")
```

### Sample output

```text
Original    Add       Delete    Result
-----------------------------------------------
play        s         -         plays
play        ed        -         played
play        ing       -         playing
playing     -         ing       play
played      -         ed        play
happy       ness      y         happiness
unhappy     -         un        happy
```

**Explanation:**

- **Add:** Adds a suffix or prefix.
- **Delete:** Removes an affix or replaces part of a word.
- For example, `play + ing = playing`.
- For `happy → happiness`, the final `y` is replaced by `i` before adding `ness`.

The table represents illustrative rules, not a full automatic morphological engine.

---
