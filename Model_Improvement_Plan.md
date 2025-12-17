# 🔧 Model Improvement Plan (For Final Report)

This section outlines the planned methodological improvements for the next phase of the project.

These upgrades directly address the empirical limitations observed in the midterm results and incorporate feedback from instructors and peers.

---

## 1. Redesigning the Popularity Score (Target Variable)

The current raw popularity formula

```
views + 5 × likes + 10 × comments
```

was useful for early exploration, but it is not statistically grounded.

### Planned improvements:
- Test alternative engagement metrics (e.g., like/view ratio, comment/view ratio).
- Apply log-transformation on raw counts to reduce long-tail skew.
- Use regression-based or data-driven weighting instead of fixed weights.
- Consider time-adjusted popularity (views per hour since upload).

---

## 2. Category-Aware Modeling

Instructor feedback highlighted that popularity dynamics differ substantially across genres.

### Planned improvements:
- Train separate models for individual categories (e.g., comedy, travel, education).
- Alternatively, use category embeddings in BERT or one-hot vectors in classical models.
- Evaluate cross-category generalization to quantify domain transfer performance.

---

## 3. Handling Long-Tail Distribution

Both views and engagement statistics follow a heavy-tailed distribution, which contributed to low R² in baseline regressions.

### Planned improvements:
- Log-scale the target variable or use quantile regression.
- Experiment with robust loss functions (e.g., Huber, Tukey) for tree-based models.
- Rebalance popularity classes to reduce dominance of low-engagement videos.

---

## 4. Incorporating Temporal Dynamics (Follow-up Crawl Data)

Popularity is inherently time-dependent, but current models treat video performance as static.

### Planned improvements:
- Use follow-up crawl data to derive growth metrics (e.g., early velocity, decay rate).
- Build simple time-series models to predict future popularity trajectories.
- Compare temporal vs. pre-upload predictors to identify timing effects.

---

## 5. Enhancing Text Modeling with Advanced Transformers

DistilBERT already outperforms all classical baselines, but its configuration can be improved.

### Planned improvements:
- Extend input sequence length to capture longer titles or richer hashtags.
- Fine-tune hyperparameters (learning rate, batch size, epochs).
- Experiment with alternative transformer architectures (BERT-base, RoBERTa).
- Apply class weights to address imbalanced popularity labels.

---

## 6. Improved Fusion of Title and Hashtag Information

The current fusion method simply concatenates titles with hashtag strings.

### Planned improvements:
- Create separate BERT embeddings for titles and hashtags.
- Combine them using an attention-based fusion layer.
- Compare simple concatenation vs. dual-encoder architectures.

---

## 7. Model Stability and Robustness Evaluation

Current evaluation relies on a single train/test split, which does not guarantee stability.

### Planned improvements:
- Implement 5-fold or 10-fold cross-validation.
- Repeat experiments with multiple random seeds and report variance.
- Evaluate model robustness across different subset sizes and categories.

---

## Implementation Priority

### High Priority (Address Core Limitations):
1. **Redesigning Popularity Score** - Foundation for all models
2. **Model Stability Evaluation** - Essential for credible results
3. **Handling Long-Tail Distribution** - Directly addresses low R² issue

### Medium Priority (Enhance Performance):
4. **Category-Aware Modeling** - Addresses instructor feedback
5. **Enhancing Text Modeling** - Builds on current best model (DistilBERT)
6. **Temporal Dynamics** - Leverages available follow-up data

### Lower Priority (Advanced Features):
7. **Improved Fusion Architecture** - More complex, may have diminishing returns

---

## Notes

- Current baseline uses: `popularity_raw = 0.65*views + 0.25*views_diff + ...` (from Modeling.ipynb Cell 2)
- DistilBERT currently uses max_length=64 for titles, max_length=128 for title+hashtag
- Follow-up data available in `data/followup_all.csv` and `followups/Follow_up/` directory
- Current evaluation uses single train/test split (90/10)

