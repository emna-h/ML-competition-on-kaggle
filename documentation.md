# INSAT MPI track prediction : full explanation of `solution.py` and the approach behind it

---

## 0. Summary in 10 lines

- **Task:** predict one of 5 classes (`GL`, `RT`, `IIA`, `IMI`, `fail`) for MPI students at INSAT. Metric: **macro-F1** (the five classes weigh the same).
- **Data:** 3,099 labelled students (cohorts 2019–2024) and 987 test students (cohorts 2025 and 2026).
- **Validation:** *year-grouped* cross-validation — whole cohort years are held out, because the test is made of future years.
- **Final model:** the model is given **no average column and no GL/RT/IIA/IMI score formula**. It gets raw grades (outliers capped at 17), a few metadata columns and a grade-format regime flag.
- **Key trick:** a first-stage logistic regression *learns* a score for each class. Those learned scores are turned into **rank inside the cohort** and **rank inside the cohort + bac section** (the competition for seats). Then six models are averaged.
- **Result:** year-grouped CV macro-F1 **0.6717**, accuracy **0.6686**.

---

## 1. The problem and the data

### 1.1 Files
| File | Rows | Columns | Content |
|---|---|---|---|
| `train.csv` | 3,099 | 40 | features + `target` |
| `test.csv` | 987 | 39 | same features, no `target` |
| `sample_submission.csv` | 987 | 2 | format: `id,target` |

Train cohorts: 2019, 2020, 2021, 2022, 2023, 2024. Test cohorts: 2025 and 2026 (entirely unseen years).

### 1.2 Columns
- **Identity / noise:** `id`, `birth_day_month`, `zodiac`, `exam_room`, `governorate`
- **Cohort:** `cohort_year`
- **Student attributes:** `gender`, `bac_section` (math / sc / info / tech), `bac_avg`, `attendance`, `club`, `redoublant`, `transfer`, `laptop_owned`, `has_github`, `knows_the_hood`
- **22 grades.** Semester 1: `an1, alg1, circ, opt, estat, meca, algo1, prog1, fr1, en1, droit`. Semester 2: `an2, alg2, syslog, electronique, thermo, emag, algo2, prog2, fr2, en2, eco`
- **`S1_rounded`:** a precomputed (rounded) semester-1 average
- **Target:** `GL`, `RT`, `IIA`, `IMI`, `fail`

Train class counts: GL 700, RT 687, IIA 635, IMI 575, fail 502.

### 1.3 Metric
**Macro-F1** = the plain average of the five per-class F1 scores. Accuracy alone is not enough, because a model could do well on big classes and poorly on small ones.

---

## 2. What the data analysis showed

| Finding | Evidence | Consequence |
|---|---|---|
| `S1_rounded` = a **weighted** S1 average (weight 1.5 for the 8 core subjects, 1 for `fr1`, `en1`, `droit`), rounded. | Reproduces it exactly (100% within rounding). The official INSAT coefficients match only 87%. | The dataset's weights are simplified, not the official table. |
| **All-zero rows = dropouts**, and they always fail. | Zeros counted per row. | Raw grades are kept as they are (zeros are real information). |
| An average ≥ 10 **never** fails; below 10, the fail probability falls smoothly (about 96% below 7, about 2% at 9.75–10). | Fail rate by average bin. | Pass/fail below 10 depends on a **resit outcome that is not in the data**. This limits the ceiling. |
| **Two grade-format regimes.** Odd cohorts (2019, 2021, 2023, 2025) use quarter-point grades; even cohorts (2020, 2022, 2024, 2026) use two-decimal grades. | Fraction of grades that are not multiples of 0.25, by cohort. | The pass/fail rule differs by regime (e.g. average 8.5–9: ~48% fail in quarter-point cohorts vs ~9% in two-decimal cohorts). The code adds a regime flag `fmt`. |
| `bac_section` is the **strongest preference proxy**. | Top-quartile math students take GL 89% of the time; tech → IIA 51%. | It is kept, one-hot encoded. |
| **GL is not a hard "top-N" by any score.** | In every cohort, 98–182 non-GL students out-score the lowest GL student. | Hidden student preferences exist → the ceiling is structural. |
| **Outliers:** some students have whole profiles at 17–19. | 26 train students with average ≥ 17. | Grades are **capped at 17** (measured gain +0.005, i.e. noise-level; kept because it fits domain knowledge). |
| **Isolated zeros are informative.** | 142 otherwise normal students have 1–3 zero grades; 119 of them failed. | Zeros are **not** treated as missing. |
| `id`, row order, birth date, exam room carry **no signal**. | AUC ≈ 0.5 for every class. | These columns are dropped. |
| No student appears across cohorts. | A strict key (bac_avg + birth date + governorate + gender) matched only 4 pairs, which is chance. | No leak to exploit, nothing to link. |

### 2.1 The two hints 
1. *Track allocations:* students do not always end up in the track they were "expected" to take → **hidden preferences + seat competition**.
2. *Score formulas in this dataset differ from the standard INSAT formulas* → the official formula cannot be trusted as-is. 
3. it is kind of useless to add a score column (GL, RT, IIA…) or a moyenne column, and especially to hand the model the formula, because the model is the one that learns the moyenne and the score."

Testing hint 3:
- **Logistic regression** is unaffected by removing average/score columns (0.656–0.661 either way) → it learns averages by itself.
- **Tree models** lost about 0.01–0.02 when *all* formula-derived columns were removed. The reason: the removed columns included **cohort-relative ranks**, which a model that sees one row at a time cannot build from raw grades.
- Fix: let a model **learn the score first**, then rank students by it within their cohort. This recovered the loss with **no hand-written formula** 

---

## 3. Validation strategy (why not a random split)

The test set is made of **future cohorts**. A random split lets the same cohort appear in train and validation, so cohort-level behaviour (pass rates, grading style) leaks and the score looks better than it will be.

**Used here:** `GroupKFold(6)` with `cohort_year` as the group → each fold holds out **one whole year**. Everything that is learned from labels (the stage-1 scores, the models) is learned **only on the training years of that fold**.

Noise level: with 3,099 rows, the standard error of macro-F1 is about **0.008–0.009**. Differences smaller than about 0.005 cannot be told apart.

---

## 4. The code:

### 4.1 Header, imports and constants
```python
CLASSES = ["GL", "RT", "IIA", "IMI", "fail"]      # class index 0..4 used everywhere
S1 = [...11 semester-1 grades...]; S2 = [...11 semester-2 grades...]
GRADES = S1 + S2                                   # the 22 raw grades
GRADE_CAP = 17.0                                   # outlier cap
BIN = ["club","redoublant","transfer","laptop_owned","has_github","knows_the_hood"]
```
`HERE` is the folder of the script, so `train.csv` and `test.csv` are found next to it (this fixed the first `FileNotFoundError` you hit, which came from a path that only existed in my sandbox).

### 4.2 `prepare_features(df)` — builds the feature table
It is called once on train and once on test. It returns `(X, cohort_year array, bac_section array)`.

1. **Regime flag `fmt`.** Computed from the *raw* grades (before any capping): a student gets `1` if any grade is not a multiple of 0.25, else `0`.
   ```python
   fmt = ((~np.isnan(g0)) & (np.abs(g0*4 - np.round(g0*4)) > 1e-6)).any(axis=1)
   ```
   This uses only the student's own grades — no labels — and exists for test cohorts too.
2. **Cleaning.**
   - `gender` → upper-case; `bac_section` → lower-case, with `technique→tech`, `sciences→sc`, `informatique→info`.
   - Yes/no columns are mapped with a dictionary (`yes/no/true/false/oui/non/1/0`). **Missing values stay missing** (they are median-imputed later).
3. **Feature table `X`:**
   - the 22 grades, **clipped at 17** (NaNs stay NaN);
   - `bac_avg`, `attendance`, the 6 yes/no columns, `fmt`;
   - `bac_avg_rk` = percentile of `bac_avg` **inside the student's cohort** (computed with `groupby("cohort_year")`, so the two test cohorts are ranked separately, not mixed);
   - one-hot encoding of `bac_section` and `gender` (no fake ordering, unlike `OrdinalEncoder`).
4. **Deliberately absent:** `S1_rounded`, any average (S1, S2, MG), any track score, `governorate`, `birth_day_month`, `zodiac`, `exam_room`, `id`, `cohort_year` itself (it cannot extrapolate to 2025–2026).

Result: **38 base features**.

### 4.3 `_lr()` — the learner used for stage 1 and as a model
`StandardScaler` → `LogisticRegression(C=0.3)`. The strong regularisation (`C=0.3`) keeps the learned score stable on 3k rows.

### 4.4 `_score_feats(S, coh, sec)` — from scores to *competition* features
`S` is an `n × 5` matrix of class probabilities. For each class `c` it creates three columns:
- `S_p_c` — the learned probability (the "learned score");
- `S_rk_c` — the percentile rank of that probability **among students of the same cohort**;
- `S_rks_c` — the percentile rank **among students of the same cohort and the same bac section**.

That is **15 new features**. Why ranks? Seats are limited per track per cohort: what matters is not only "how good is this student" but "how many students rank above them". A per-row model cannot compute that from raw grades.

### 4.5 `add_learned_scores(Xa, ya, ga, sa, Xb, gb, sb)` — nested stacking, **no leakage**
- **Rows of A (training):** scores are **out-of-fold**. `GroupKFold(min(5, #cohorts))` splits by cohort year; for each split, a logistic regression is fitted on the other years and predicts the held-out year.
- **Rows of B (the rows to predict):** a logistic regression fitted on all of A predicts them. B's labels are never used.
- Then `_score_feats` is applied to each side with its own cohort / section arrays.

Why out-of-fold? If the training scores came from a model fitted on those same rows, they would be over-optimistic and the second-stage models would learn to trust them too much.

### 4.6 `SeedAvg` — averaging several random seeds
A small wrapper that fits the same model with seeds `(0, 1, 2)` and averages `predict_proba`. It lowers the variance of the tree models. It drops its factory lambda after fitting so the object can be pickled.

### 4.7 `make_models()` — the six models
| Model | Main settings |
|---|---|
| `logreg` | standardised, `C=0.3` |
| `hgb` (HistGradientBoosting) | lr 0.05, 200 iters, depth 3, L2 5, min leaf 30 |
| `extratrees` | 500 trees, min leaf 3, `max_features=0.5`, 3 seeds |
| `lgbm` | 300 trees, lr 0.03, 8 leaves, subsample 0.8, colsample 0.6, λ=5, 3 seeds |
| `xgb` | 300 trees, lr 0.03, depth 3, subsample 0.8, colsample 0.6, λ=5, 3 seeds |
| `catboost` | 600 iters, lr 0.04, depth 5, L2 6, 1 seed |

They are **diverse** (linear, bagged trees, three boosting libraries).  keeping all six of threm because averaging them reliably beats the best single model;  testing proved that the ensemble beats each member under identical splits.

### 4.8 `main()` — the pipeline
1. **Arguments:** `python solution_1.py [data_dir] [out_dir] [--cv]`. Default: both = folder of the script.
2. **Load** `train.csv` / `test.csv`; map `target` to indices 0–4.
3. **Features** for train and test; `Xt.reindex(columns=X.columns, fill_value=0.0)` guarantees the same columns in the same order (one-hot columns can differ).
4. **Median imputation** fitted on train (labels not used) and applied to both.
5. **`--cv` block (optional):** for each of the 6 year-folds → `add_learned_scores` on that fold only → fit every model → store out-of-fold probabilities. Then print per-model macro-F1, the **ensemble** macro-F1 (plain average of probabilities, `argmax`), accuracy, the per-class report and the confusion matrix.
6. **Final fit:** `add_learned_scores` on **all** labelled data (out-of-fold for train, fitted for test) → fit each of the 6 models → average the 6 probability matrices → `argmax`.
7. **Submission:** `DataFrame({id, target})` with the original test IDs in the original order; `assert` checks the row count, ID uniqueness and that every label is valid; saved as `submission.csv`.

**Why plain `argmax`?**  testing alternatives on the out-of-fold probabilities: per-class decision weights (nested), EM prior adaptation per cohort, and capacity-matched assignment. All were equal or worse, so the simplest rule stays.

---

## 5. Results (all measured, year-grouped CV)

### 5.1 This solution
| Model | Macro-F1 |
|---|---|
| logreg | 0.6574 |
| hgb | 0.6642 |
| extratrees | 0.6621 |
| lgbm | 0.6706 |
| xgb | 0.6768 |
| catboost | 0.6687 |
| **Ensemble (the 6, averaged)** | **0.6717** (accuracy 0.6686, log-loss 0.8347) |

Per-class F1 of the ensemble: **GL 0.760, RT 0.609, IIA 0.629, IMI 0.506, fail 0.854**.

By held-out year: 2019 0.675 · 2020 0.684 · 2021 0.675 · 2022 0.656 · 2023 0.662 · 2024 0.675.

Confusion matrix (rows = actual):

| | GL | RT | IIA | IMI | fail |
|---|---|---|---|---|---|
| **GL** | 538 | 82 | 54 | 26 | 0 |
| **RT** | 83 | 405 | 95 | 104 | 0 |
| **IIA** | 55 | 69 | 408 | 92 | 11 |
| **IMI** | 39 | 85 | 96 | 290 | 65 |
| **fail** | 0 | 3 | 9 | 59 | 431 |

IMI is the weakest class: it behaves like the "didn't get my choice" destination, and it is confused with RT, IIA and fail.


### 5.2 What has been tested that did **not** help
Each was measured under the same year-grouped splits and was within noise or worse:
- Official INSAT coefficients, the four official track scores, and unit (UE) averages.
- Searching free coefficients for sharper GL/RT cutoffs (better on training cohorts, no better on held-out cohorts).
- Dropping extreme training rows, winsorising, other caps (16 / 18), zeros-to-missing.
- Governorate target encoding (nested), resit-effort deficit features, absolute seat-pressure counts, expected-seat-consumption features.
- Splitting the learned score into a "merit" part (grades only) and a "preference" part (metadata only).
- Section-specific learned weights, section-relative `bac_avg` / attendance.
- Hierarchical model (fail first, then track among passers).
- Dropping noise-like binary columns.
- Per-class decision weights, EM prior adaptation, capacity-matched assignment.

---

## 6. The submission file `submission.csv`

| Check | Result |
|---|---|
| Rows | 987 (= test rows) |
| Columns | `id,target` (matches `sample_submission.csv`) |
| IDs | unique, same values and same order as `test.csv` |
| Labels | all in {GL, RT, IIA, IMI, fail}; none missing |
| Class counts | GL 233 · IIA 205 · RT 199 · IMI 181 · fail 169 |

By cohort, predicted `fail` share is 20.1% for 2025 (a quarter-point cohort) and 14.2% for 2026 (a two-decimal cohort), consistent with the regime difference seen in training (≈19% vs ≈13%).

---

## 7. How to run it

```bash
pip install numpy pandas scikit-learn lightgbm xgboost catboost
python solution.py            # writes submission.csv next to the script
python solution.py --cv       # also prints the year-grouped validation report (a few minutes)
python solution.py DATA_DIR OUT_DIR   # custom folders
```

---

## 9. Glossary

- **Cohort:** all students of one academic year (`cohort_year`).
- **Year-grouped CV:** cross-validation where all rows of a cohort stay together in the held-out fold.
- **Out-of-fold (OOF):** a prediction made for a row by a model that never saw that row.
- **Stacking:** feeding the predictions of a first-stage model to a second-stage model.
- **Percentile rank:** the fraction of students with a lower value, inside a group (here: cohort, or cohort + section).
- **Macro-F1:** the average of the five per-class F1 scores.
- **Log-loss:** measures the quality of the predicted probabilities, not just the final label; lower is better.
