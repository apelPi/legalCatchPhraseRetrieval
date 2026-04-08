# Technical Architecture & Algorithm Documentation

This document explicitly details the software architecture, scripts, and rigorous algorithmic theory driving the Information Retrieval and Extraction (IRE) project focused on legal domain catchphrase detection.

## 1. Codebase and Script Integration
The core workflow surrounds analyzing legal cases to harvest defining phrases (catchphrases) without manual annotation. Here is how our components interact:

### A. Orchestration Layer (`evaluate_catchphrases.py`)
This custom script operates as the command center for the experiment, bridging multiple Python packages and algorithms.

1. **Environmental Setup and Negative Baseline Construction:** 
   The script invokes `sklearn.datasets.fetch_20newsgroups` to pull down records from the `20 Newsgroups` dataset natively into memory, stripping out email headers, footers, and quotes (`remove=('headers', 'footers', 'quotes')`). This cleans the text so that it mimics pure conversational English. The text is dumped into a `nonlegal_docs` directory acting as the out-of-domain control corpus.
   
2. **Model Instantiation and Training (`pslegal.py`):**
   The script instantiates the primary class `pvect = psl.PSlegalVectorizer(version=1)`. Rather than fitting directly on in-memory arrays which could hit RAM ceilings on 18,000+ files, it uses `pvect.efficient_fit(legal_dir, nonlegal_dir, gram='nnp')`. Internally, `efficient_fit` walks through the directories file by file, computing collection frequencies (`CF`) and document frequencies (`DF`) for both corpora asynchronously using Python dictionaries.

3. **TF-IDF Baseline Construction (`TfidfVectorizer`):**
   To produce a competing benchmark, the script loads all `Train_docs` into an array and fits `sklearn.feature_extraction.text.TfidfVectorizer(stop_words='english', ngram_range=(1, 4))`. This computes the traditional inverse document frequency weights without cross-domain contrast.

### B. Extraction Layer (`extract_noun_phrases.py`)
This script acts as the structural foundation of the parser. When `evaluate_catchphrases.py` evaluates a `Test_doc`, it first passes the raw text to `extract_noun_phrases.extract(text)`.
1. **Sentence Boundary Disambiguation:** Uses `my_tokenizer1()` to intelligently tokenize sentences based on trailing periods, shielding against ellipses and abbreviations.
2. **Regex Chunking (`nltk.RegexpParser`):** It applies multiple grammars against POS-tagged sentences. For example, the regex `{<JJ|VBD|VBN|VBG><NN|NNS>}` triggers matches for adjoining Adjectives/Past-tense Verbs and Nouns.
3. **Filtering & Post-Processing:** The script maps the extracted clusters against `legal_specific_stopwords.txt` and dynamically removes leading bounding determiners (like 'the', 'an') mapped against our custom `list_of_determiners_from_wikipedia_page.txt`.

### C. Execution and Metric Validation
For each document in `Test_docs`, `evaluate_catchphrases.py`:
1. Passes the raw text through `extract_noun_phrases.py` to yield candidate noun phrases (NNPs).
2. Scores each candidate NNP string independently via `pvect.get_score([nnp])`, logging internal `KeyError` exceptions silently if the n-gram was omitted during training bounds.
3. Queries the `TfidfVectorizer` by matching the exact same NNPs against the TF-IDF feature matrix (`tfidf.vocabulary_.get(w)`), summing term weights.
4. Retrieves the top-15 ranked phrases produced by both regimes.
5. Computes Precision, Recall, and F1 by executing a case-insensitive set-intersect against parsed `Test_catches` (the human-curated gold standard).

---

## 2. In-Depth Algorithm Analysis

Legal catchphrase extraction can be modeled in two radically different paradigms: Unsupervised and Supervised.

### A. The Unsupervised Approach (`pslegal`)
The `pslegal` architecture attempts to extract catchphrases using a blend of deterministic grammatical chunking and contrastive frequency analysis.

#### Step 1. Grammatical Parsing
Catchphrases rarely contain active verbs; they are almost universally defined as noun phrases (NP). Using NLTK (Natural Language Toolkit), the algorithm chunks tokens targeting strict POS tag formations. For example, it searches for consecutive sequences of Nouns (`NN/NNS/NNP`) connected dynamically by Prepositions (`IN`) or Adjectives (`JJ`), discarding stopwords. 

#### Step 2. Legal Domain Importance Calculation
Once a candidate noun phrase $c$ is extracted, it undergoes the $PSLegal$ scoring. Instead of just analyzing intra-document frequency, it measures specific domain bias:
$$ Importance(t, C) = \frac{CF(t, C)}{DF(t, C) + 1} $$
Where $CF$ is the Collection Frequency and $DF$ is Document Frequency. 
The algorithm calculates:
$$ Score(t_{legal}) = \frac{Importance(t, C_{legal})}{Importance(t, C_{nonlegal}) + 1} $$
If a sequence token $t$ appears massively in the `Court Cases` dataset but rarely in the `20 Newsgroups` dataset, its domain contrast score skyrockets. 

#### Step 3. KL-Divergence Informativeness (KLI) Modulator
To prevent generic widespread legal jargon like "court" from dominating specific unique phrases, it modulates the sequence score via KL-divergence:
$$ KLI(c,d) = \frac{TF(c,d)}{|d|} \times \log\left(\frac{TF(c,d)/|d|}{CF(c)/N}\right) $$
The final un-supervised algorithm output is a marriage of KLI and the contrasting domain sequence score.

### B. The Supervised Approach (`D2V-BiGRU-CRF`)
While `pslegal` analyzes relative math, the neural supervised model (`D2V-BiGRU-CRF`) formulates this as a **Sequence Labeling problem** (similar to Named Entity Recognition). It evaluates whether the word at index $i$ is `B-CP` (Beginning of Catchphrase), `I-CP` (Inside Catchphrase), or `O` (Outside).

#### The Architecture Pipeline:
1. **Document-to-Vector (Doc2Vec):** Generates low-dimensional dense embeddings of standard legal word sequences to trap deeply semantic correlations that standard lexical matching drops.
2. **Bidirectional Gated Recurrent Units (BiGRU):** Recurrent Neural Networks traditionally struggle with "forgetting" early contexts. A GRU captures both the long-term context that generated a word and the short-term syntax. By making it *Bidirectional*, the GRU reads the legal phrase both forwards and backwards, meaning the hidden state for a word like "Double" holds knowledge regarding the future word "Taxation".
3. **Conditional Random Fields (CRF):** Instead of making final predictions randomly word-by-word at the exit of the GRU layer, a CRF evaluates the entire prediction sequence probability. Structurally recognizing that an `I-CP` output state is highly likely if the preceding position was `B-CP`.

**The Tradeoff:**
The neural `BiGRU-CRF` approach captures incredibly deep contextual meaning resulting in substantially higher extraction capabilities. However, its significant limitation is dependency; it requires thousands of human-annotated catchphrase labels to correctly converge during initial tensor gradient descent. When training labels aren't available, the contrastive statistics of the `pslegal` architecture serve as a highly functional, scalable fallback.
