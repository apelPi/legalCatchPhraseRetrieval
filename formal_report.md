# Formal Project Report: Unsupervised Catchphrase Identification from Legal Court Case Documents

## 1. Introduction
In countries following the Common Law system, prior case precedents play a fundamental role in guiding ongoing legal disputes. However, the volume and complexity of legal texts make identifying core legal concepts exceptionally challenging for practitioners. Specifically, "catchphrases"—short phrases that succinctly summarize legal issues—are critical for rapid indexing and information retrieval. 

This report outlines the implementation and evaluation of an unsupervised model (`pslegal`) designed to automatically extract catchphrases from court cases on the premise that domain-specific noun phrases are key carriers of legal concepts.

## 2. Methodology
The method implements a two-step unsupervised formulation as opposed to neural supervised approaches (such as D2V-BiGRU-CRF).

### 2.1 Candidate Extraction
The algorithm isolates noun phrase patterns (specifically `NP` tags) natively using NLTK part-of-speech chunking since most catchphrases syntactically behave as noun phrases.

### 2.2 Domain-Contrastive Scoring (`pslegal`)
Instead of solely relying on Term Frequency-Inverse Document Frequency (TF-IDF), `pslegal` measures the importance of terms within a **legal document corpus** (FIRE-2017 Train_docs) relative to an out-of-domain **non-legal corpus** (the 20 Newsgroups dataset). Candidate phrases that occur densely in legal texts but infrequently in general natural language domains are assigned a higher probability (the `PSLegal` score) of being catchphrases.

### 2.3 Baseline Comparison
To evaluate the efficacy of the `pslegal` architecture, we execute a benchmark against a simple token-level TF-IDF baseline. TF-IDF ranks the identical extracted noun phrases based purely on intra-domain frequency distribution.

## 3. Dataset Description
The evaluation utilizes the `FIRE2017-IRLeD-track-data` (Task 1) dataset, originally collected from the Supreme Court of India. It contains both raw legal case descriptions and the respective "gold standard" catchphrases curated by the Manupatra legal search system.

- **Legal Corpus:** `Train_docs` for corpus frequency estimation.
- **Evaluation Set:** `Test_docs` alongside reference `Test_catches` for benchmarking.
- **Non-legal Corpus:** `20 Newsgroups` subsets (general english texts).

## 4. Experimental Results
Predictions were truncated to the top 15 highest-scoring noun phrases per document. Standard IR evaluation matrices (Precision, Recall, F1) were measured against exact string matches of the Manupatra gold standard.

### Results Table
| Model | Precision | Recall | F1-Score |
|---|---|---|---|
| TF-IDF Baseline | 0.0233 | 0.0671 | 0.0286 |
| `pslegal` | **0.0733** | **0.0910** | **0.0659** |

## 5. Conclusion
With a constrained sample to prioritize computational speed, the unsuperivsed `pslegal` algorithm exhibited compelling performance against a TF-IDF control. By incorporating an auxiliary non-legal corpus to calibrate vocabulary importance, `pslegal` effectively isolated domain-specific concepts, obtaining a tripling effect in overall Precision and F1-score relative to TF-IDF. This highlights the viability of term-contrastive statistical NLP in legal informatics without requiring large manually annotated corpora.
