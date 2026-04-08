import os
import glob
import pslegal as psl
import extract_noun_phrases as npe
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import shutil

base_dir = r"c:\Users\adidw\Documents\courseStuff\sem8\ireProj"
legal_train_dir = os.path.join(base_dir, "FIRE2017-IRLeD-track-data", "Task_1", "Train_docs")
test_docs_dir = os.path.join(base_dir, "FIRE2017-IRLeD-track-data", "Task_1", "Test_docs")
test_catches_dir = os.path.join(base_dir, "FIRE2017-IRLeD-track-data", "Task_1", "Test_catches")
nonlegal_dir = os.path.join(base_dir, "nonlegal_docs")

# Create mini dirs for faster training
legal_train_mini = os.path.join(base_dir, "legal_train_mini")
nonlegal_mini = os.path.join(base_dir, "nonlegal_mini")
os.makedirs(legal_train_mini, exist_ok=True)
os.makedirs(nonlegal_mini, exist_ok=True)

# Copy 10 legal and 50 non-legal docs
for i, f in enumerate(glob.glob(os.path.join(legal_train_dir, "*.txt"))):
    if i >= 10: break
    shutil.copy(f, os.path.join(legal_train_mini, os.path.basename(f)))
    
for i, f in enumerate(glob.glob(os.path.join(nonlegal_dir, "*.txt"))):
    if i >= 50: break
    shutil.copy(f, os.path.join(nonlegal_mini, os.path.basename(f)))

print("Training pslegal vectorizer (Mini version)...")
pvect = psl.PSlegalVectorizer(version=1)
pvect.efficient_fit(legal_train_mini, nonlegal_mini, gram='nnp')

print("Training TF-IDF baseline...")
train_docs = glob.glob(os.path.join(legal_train_dir, "*.txt"))
tfidf_texts = []
for p in train_docs:
    with open(p, "r", encoding="utf-8", errors='ignore') as f:
        tfidf_texts.append(f.read())
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 4))
tfidf.fit(tfidf_texts)

pslegal_precisions, pslegal_recalls, pslegal_f1s = [], [], []
tfidf_precisions, tfidf_recalls, tfidf_f1s = [], [], []

def compute_metrics(predicted, gold):
    if not predicted or not gold: return 0, 0, 0
    pred_set = set([p.lower().strip() for p in predicted])
    gold_set = set([g.lower().strip() for g in gold])
    
    true_positives = len(pred_set.intersection(gold_set))
    precision = true_positives / len(pred_set) if len(pred_set) > 0 else 0
    recall = true_positives / len(gold_set) if len(gold_set) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

# Limit test evaluation to 20 documents for speed
test_files = glob.glob(os.path.join(test_docs_dir, "*.txt"))[:20]
print(f"Evaluating on {len(test_files)} test documents...")
for test_file in test_files:
    case_name = os.path.basename(test_file).replace("_statement.txt", "")
    catch_file = os.path.join(test_catches_dir, case_name + "_catchwords.txt")
    
    if not os.path.exists(catch_file):
        continue
        
    with open(catch_file, "r", encoding="utf-8", errors="ignore") as f:
        gold_catches = []
        for line in f:
            gold_catches.extend([c.strip() for c in line.split(',') if c.strip()])
    if not gold_catches: continue
    
    with open(test_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    nnps = npe.extract(text)
    pvect.fit_doc(nnps)
    scored_nnps = []
    seen_nnps = set()
    for nnp in nnps:
        nnp_str = nnp.lower().strip()
        if nnp_str not in seen_nnps:
            try:
                score = pvect.get_score([nnp])
                scored_nnps.append((score, nnp_str))
                seen_nnps.add(nnp_str)
            except Exception:
                pass
                
    scored_nnps.sort(reverse=True)
    pslegal_pred = [x[1] for x in scored_nnps[:15]]
    
    vector = tfidf.transform([text])
    tfidf_scores = []
    for nnp in seen_nnps:
        words = nnp.split()
        score = sum([vector[0, tfidf.vocabulary_.get(w)] for w in words if tfidf.vocabulary_.get(w) is not None])
        tfidf_scores.append((score, nnp))
    tfidf_scores.sort(reverse=True)
    tfidf_pred = [x[1] for x in tfidf_scores[:15]]

    pp, pr, pf1 = compute_metrics(pslegal_pred, gold_catches)
    tp, tr, tf1 = compute_metrics(tfidf_pred, gold_catches)
    
    pslegal_precisions.append(pp)
    pslegal_recalls.append(pr)
    pslegal_f1s.append(pf1)
    
    tfidf_precisions.append(tp)
    tfidf_recalls.append(tr)
    tfidf_f1s.append(tf1)

print("\n--- RESULTS ---")
res_pslegal = f"PSLegal - Avg Precision: {np.mean(pslegal_precisions):.4f}, Avg Recall: {np.mean(pslegal_recalls):.4f}, Avg F1: {np.mean(pslegal_f1s):.4f}"
res_tfidf = f"TFIDF   - Avg Precision: {np.mean(tfidf_precisions):.4f}, Avg Recall: {np.mean(tfidf_recalls):.4f}, Avg F1: {np.mean(tfidf_f1s):.4f}"
print(res_pslegal)
print(res_tfidf)

with open(os.path.join(base_dir, "results.txt"), "w") as f:
    f.write(res_pslegal + "\n" + res_tfidf + "\n")
