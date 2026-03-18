# Spam Classifier Thinking (Design Notes)

## Goal
Detect whether an incoming message/email is **spam** or **ham** (not spam) with high **precision** (avoid false positives) while maintaining good **recall** (catch most spam).

## Data needed
- **Labeled examples**: message text + label (`spam`/`ham`), ideally from the same channel you’ll deploy on (SMS vs email vs in-app chat differ a lot).
- **Metadata (optional but useful)**, depending on privacy/availability:
  - Sender info (domain, reputation, first-seen time, whether in contacts)
  - Message route (inbound source, gateway, IP / ASN if applicable)
  - Time features (hour/day; bursty senders)
  - Link destinations (final URL domain category, redirect depth)
- **Feedback loop**: user “mark as spam / not spam” corrections for continuous improvement and drift handling.

## Feature ideas
### Text features (high-signal, cheap)
- **TF‑IDF word n‑grams** (1–2): common spam phrases (“win”, “free”, “limited time”).
- **TF‑IDF character n‑grams** (3–5): robust to obfuscation (“fr€e”, “w1n”, “c1ick”).

### Lightweight heuristic stats (good complement to TF‑IDF)
- Message length, word count
- URL count (`http`, `https`, `www`)
- Email/phone number patterns
- Digit ratio, uppercase ratio
- Punctuation intensity (`!!!`, repeated symbols)
- Keyword flags (e.g. “unsubscribe”, “urgent”, “verify”, “OTP” depending on domain)

### Advanced / non-text (if you have it)
- Sender/domain reputation features
- Graph features (sender–recipient relationship, contact network)
- URL reputation / category features
- Rate features (same sender blasting many recipients)

## Model choices (practical)
- **Linear models** (Logistic Regression / Linear SVM): strong baseline for text spam filtering, fast, interpretable.
- **Naive Bayes**: very fast baseline, often competitive for bag-of-words.
- **Tree/boosting**: good when you have many non-text features; careful with sparse text.
- **Deep learning**: useful at scale, but more costly and easier to overfit without enough data.

## Evaluation (what to measure)
- Prefer **precision/recall/F1** over accuracy (spam is often imbalanced).
- Use a **confusion matrix** and tune a **decision threshold** to match your product risk:
  - If false positives are costly, raise the threshold for labeling spam.
- Keep a **time-based holdout** (train on older, test on newer) to detect drift.

## Common mistakes / failure modes
- **Data leakage**: features that “peek” at the label (e.g., using user action fields that are consequences of classification).
- **Train/test contamination**: duplicates across splits; near-duplicates are common in spam campaigns.
- **Wrong metric**: optimizing accuracy while precision collapses.
- **Domain mismatch**: training on SMS but deploying on email; performance drops.
- **Concept drift**: spam tactics change; models decay without retraining.
- **Bias and fairness issues**: aggressive rules can harm specific languages/dialects or legitimate marketing.
- **Over-cleaning text**: stripping punctuation/characters can remove key spam signals.
- **Threshold not calibrated**: using the default 0.5 without aligning to business cost.

## Minimal system design (deployment)
1. **Ingest** message + metadata.
2. **Feature extraction** (text + optional metadata).
3. **Model scoring** → probability of spam.
4. **Thresholding + rules** (optional): allowlists/denylists, sender reputation.
5. **Action**: inbox vs spam folder vs quarantine.
6. **Logging + feedback**: collect corrections; monitor drift; retrain periodically.

