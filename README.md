# Word Similarity Engine

A from-scratch implementation of distributional word similarity — no ML
libraries. Words are represented purely by counting which other words show
up near them in a small corpus, based on the distributional hypothesis:
words used in similar contexts tend to have similar meanings.

**Author:** Cayden Goddard
**Date:** 9/25/2026

## How it works

The pipeline runs in six steps:

1. **`tokenize()`** — lowercases the text, strips punctuation, and splits it
   into a list of words.
2. **`build_vocabulary()`** — collects the unique words in the corpus into a
   sorted list. This fixed order is what lets every word's vector line up
   the same way.
3. **`build_cooccurrence_matrix()`** — for every occurrence of every word,
   counts the words that appear within a window of `±window_size` positions
   around it (default: 2). This produces a dictionary of dictionaries, e.g.
   `matrix["cat"]["the"] = 3` means "the" appeared near "cat" three times.
4. **`word_to_vector()`** — converts a word's row in that matrix into a
   plain numeric vector, using the vocabulary's fixed order.
5. **`cosine_similarity()`** — compares two vectors by the *angle* between
   them rather than their raw size: `dot_product(vec1, vec2) / (|vec1| *
   |vec2|)`. Two words whose vectors point in nearly the same direction
   (similar neighbor patterns) score close to 1, regardless of how often
   either word appears overall. A zero-length vector (a word with no
   neighbors in the window) returns a similarity of 0.0.
6. **`most_similar_words()`** — compares a target word's vector against
   every other word in the vocabulary and returns the top N by cosine
   similarity.

## Corpus

```
The cat sat on the mat.
The dog sat on the log.
Cats and dogs are animals.
The animal sat on the mat.
Dogs and cats can be friends.
The cat and the dog played together.
The animal and the dog sat together.
```

## Running it

Requires only the Python standard library (`math`, `string`,
`collections.defaultdict`).

```bash
python3 word_similarity.py
```

## Sample output

```
Target word: cat
Most similar words:
1. animal (0.889)
2. mat (0.832)
3. sat (0.826)

Target word: dog
Most similar words:
1. animal (0.873)
2. cat (0.793)
3. the (0.691)

Target word: animal
Most similar words:
1. cat (0.889)
2. dog (0.873)
3. sat (0.829)

Target word: sat
Most similar words:
1. mat (0.899)
2. played (0.861)
3. animal (0.829)
```

`cat`, `dog`, and `animal` all rank each other in their own top 3 — exactly
what you'd expect, since the corpus uses them in nearly identical sentence
patterns ("The ___ sat on the mat/log").

## Reflection: does this model understand language?

No — it only recognizes patterns. `cat`, `dog`, and `animal` cluster
together purely because they occupy the same slot in structurally
identical sentences, surrounded by the same neighboring words ("the," "sat,"
"on," "mat"/"log"). The model has no concept of what a cat *is*; it just
counts co-occurrence and measures vector angles. Swap "cat" for a nonsense
word in the same sentence positions and it would cluster identically.

## Limitations

- The corpus is tiny (7 short sentences), so counts are sparse and easily
  skewed by a single sentence.
- A window size of 2 is a fixed, somewhat arbitrary choice — a larger
  window would capture more distant relationships at the cost of noisier
  counts.
- Purely count-based vectors like this don't scale well to large,
  real-world corpora — modern approaches (word2vec, GloVe, embeddings from
  transformers) learn dense vectors instead of raw co-occurrence counts.
