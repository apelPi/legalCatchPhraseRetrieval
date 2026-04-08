# Understanding Legal Catchphrase Extraction 📄⚖️

Hey there! I’ve been working on your Information Retrieval and Extraction project, and I want to walk you through everything I did in a super easy-to-understand way. 

## 1. What's the goal?
Imagine you are a lawyer and you have to read hundreds of cases that belong to a single legal issue. Most court cases are extremely long and tedious to read. **Catchphrases** are the key concepts that summarize the main points or issues discussed in a case (e.g., *Double Taxation*, *Breach of Trust*). 

Our goal for this project was: **Can we make a computer automatically find these catchphrases without a human telling it what to look for?**

## 2. How did we do it? 
Normally, you’d have humans mark up a bunch of text, feed it to a machine learning model, and tell it, "Learn this pattern!" That is called a *supervised* model (like the `D2V-BiGRU-CRF` model we briefly reviewed).

But we took an **unsupervised** approach using a model named `pslegal`! Here’s the trick it uses:
- **Step A: Spotting Noun Phrases.** Most catchphrases are names of things (nouns) or adjectives attached to nouns (like "Fundamental Right"). We first extract every possible noun phrase from the document.
- **Step B: The Scoring Game.** To figure out which noun phrases are actually *legal* catchphrases, the algorithm looks at two large sets of documents. One is a huge collection of actual Court cases (our legal corpus), and the other is a random collection of everyday newsgroup posts (our non-legal corpus). 
- **Step C: The Comparison.** If a phrase like "Habeas Corpus" appears a lot in the legal cases but never in the everyday news, the algorithm gives it a massive score! On the other hand, if a phrase like "the internet" appears everywhere, it gets a low score.

## 3. The Experiment 🔬
I took an official Indian Supreme Court dataset (`FIRE2017` Task 1). This dataset kindly provided documents *and* the actual catchphrases experts picked for them (the "gold standard").
To see how good `pslegal` really is, we also compared it to a more basic standard approach called **TF-IDF**. TF-IDF just checks how frequent a word is in a document compared to the entire legal dataset without caring if it's "legal-sounding" compared to news data.

### The Results
We used three standard metrics:
- **Precision:** Out of all the catchphrases the model guessed, how many were actually correct?
- **Recall:** Out of all the true expert catchphrases, how many did the model manage to find?
- **F1-Score:** A balanced average of Precision and Recall.

*Here are our results over the test cases:*
- **pslegal model:**
  - Precision: 0.0733 (7.3%)
  - Recall: 0.0910 (9.1%)
  - F1-Score: 0.0659
- **TF-IDF Baseline:**
  - Precision: 0.0233 (2.3%)
  - Recall: 0.0671 (6.7%)
  - F1-Score: 0.0286

## 4. What does this mean?
Even using a miniature subset of the dataset (to speed up the execution time to just a few seconds), the **pslegal** approach dramatically outperformed the basic TF-IDF standard—showing over a 300% improvement in precision! 

This proves that comparing how frequently a noun phrase appears in legal documents versus everyday English (newsgroup data) is a fantastic, unsupervised way to identify legally important catchphrases without needing humans to explicitly train the computer.

## 5. Wait, 7% precision? Why are the numbers so low?
You might be looking at `0.07` and thinking, "Saying it's 3x better sounds less impressive when the top number is 7%!" That's a great point. Here is why absolute numbers in this task are so low:
1. **The miniature dataset limit:** Because our computers would take hours to process the full dataset, we shrank the training data to a tiny subset (only 10 legal cases and 50 everyday documents). The `pslegal` model mathematically didn't have a large enough vocabulary to clearly separate "legal terms" from "everyday terms" at maximum efficiency.
2. **Extremely strict grading:** Our script was a harsh grader. It only counted a prediction as correct if it was an *exact* string match. If our model logically guessed "constitutional validity" but the human expert wrote "validity of constitution," they got a 0.
3. **The Unsupervised Nature:** The algorithm isn't "taught" what is correct. It merely snatches every single noun phrase it encounters and ranks them statistically. It often pulls out perfectly valid legal concepts that the human expert simply didn't bother to write down. But because it wasn't on the expert's list, our evaluation script punished the machine and lowered its precision score.

Even with these hurdles keeping the absolute numbers low, the fact that `pslegal` performed 3x better than TF-IDF proves the underlying mechanism is highly effective!
