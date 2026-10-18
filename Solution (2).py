"""
solution_1.py -- FORMULA-FREE version (follows the hint: the model itself learns the moyenne and the track scores,
so we do NOT hand it any average column or any GL/RT/IIA/IMI score formula).

Pipeline
  1. features = raw grades (outliers capped at 17) + bac_avg, attendance, yes/no columns, bac section, gender,
     grade-format regime flag (cohort-level), per-cohort percentile of bac_avg.   (no averages, no formulas)
  2. stage 1: a regularised logistic regression LEARNS the 5-class score from those features
     (out-of-fold for training rows, grouped by cohort year -> no leakage; fitted model for test rows).
  3. the learned scores are turned into COHORT-RELATIVE standing: rank inside the cohort and inside
     (cohort, bac section) -- the competition-for-seats signal that a per-row model cannot build alone.
  4. stage 2: ensemble of 6 models on [features + learned scores/ranks].
  Validation = year-grouped CV: whole cohort years are held out, like the 2025/2026 test.

Usage:  python solution_1.py [data_dir] [out_dir] [--cv]
Needs:  numpy pandas scikit-learn lightgbm xgboost catboost
"""
import os, sys, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["GL", "RT", "IIA", "IMI", "fail"]
S1 = ["an1", "alg1", "circ", "opt", "estat", "meca", "algo1", "prog1", "fr1", "en1", "droit"]
S2 = ["an2", "alg2", "syslog", "electronique", "thermo", "emag", "algo2", "prog2", "fr2", "en2", "eco"]
GRADES = S1 + S2
GRADE_CAP = 17.0
BIN = ["club", "redoublant", "transfer", "laptop_owned", "has_github", "knows_the_hood"]


def prepare_features(df):
    """Returns (features, cohort_year array, bac_section array). No averages, no score formulas."""
    raw = df.copy()
    g0 = raw[GRADES].values.astype(float)
    fmt_student = ((~np.isnan(g0)) & (np.abs(g0 * 4 - np.round(g0 * 4)) > 1e-6)).any(axis=1).astype(float)
    # Grade-format regime is a COHORT property (quarter-point vs two-decimal grades). The per-student test misfires for
    # a few % of students (e.g. almost no real grades), so use the cohort-level majority. Features only, no labels.
    fmt = (pd.Series(fmt_student, index=raw.index).groupby(raw["cohort_year"]).transform("mean") > 0.5).astype(float).values
    d = raw.copy()
    d["gender"] = d["gender"].astype(str).str.upper().str.strip()
    d["bac_section"] = d["bac_section"].astype(str).str.lower().str.strip().replace(
        {"technique": "tech", "sciences": "sc", "informatique": "info"})
    bmap = {"yes": 1, "no": 0, "true": 1, "false": 0, "oui": 1, "non": 0, "1": 1, "0": 0, "1.0": 1, "0.0": 0}
    for c in BIN:   # missing stays missing (median-imputed later), it is NOT turned into "no"
        d[c] = d[c].astype(str).str.lower().str.strip().map(bmap)
    X = d[GRADES].astype(float).clip(upper=GRADE_CAP).copy()               # outliers: cap at 17
    X["bac_avg"], X["attendance"] = d["bac_avg"].astype(float), d["attendance"].astype(float)
    for c in BIN:
        X[c] = d[c]
    X["fmt"] = fmt
    X["bac_avg_rk"] = d.groupby("cohort_year")["bac_avg"].rank(pct=True)
    X["sec_"] = d["bac_section"].values
    X["gen_"] = d["gender"].values
    X = pd.get_dummies(X, columns=["sec_", "gen_"], dtype=float)
    return X, d["cohort_year"].values, d["bac_section"].fillna("NA").values


def _lr():
    return make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))


def _score_feats(S, coh, sec):
    F = {}
    for k, c in enumerate(CLASSES):
        s = pd.Series(S[:, k])
        F["S_p_" + c] = S[:, k]                                          # learned class score
        F["S_rk_" + c] = s.groupby(coh).rank(pct=True).values            # standing in the cohort
        F["S_rks_" + c] = s.groupby([coh, sec]).rank(pct=True).values    # standing among same-section students
    return pd.DataFrame(F)


def add_learned_scores(Xa, ya, ga, sa, Xb, gb, sb):
    """Rows of A (training): out-of-fold scores, folds = whole cohort years. Rows of B: scores from a model fitted on A.
    Labels of B are never used."""
    Xa, Xb = Xa.reset_index(drop=True), Xb.reset_index(drop=True)
    pa = np.zeros((len(Xa), 5))
    for a, b in GroupKFold(min(5, len(set(ga)))).split(Xa, ya, ga):
        pa[b] = _lr().fit(Xa.iloc[a], ya[a]).predict_proba(Xa.iloc[b])
    pb = _lr().fit(Xa, ya).predict_proba(Xb)
    return (pd.concat([Xa, _score_feats(pa, ga, sa)], axis=1),
            pd.concat([Xb, _score_feats(pb, gb, sb)], axis=1))


class SeedAvg:
    def __init__(self, factory, seeds):
        self.factory, self.seeds = factory, seeds

    def fit(self, X, y):
        self.ms = [self.factory(sd).fit(X, y) for sd in self.seeds]
        self.factory = None
        return self

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.ms], 0)


SEEDS = (0, 1, 2)
HGB_PARAMS = dict(learning_rate=0.05, max_iter=200, max_depth=3, l2_regularization=5.0, min_samples_leaf=30)


def make_models():
    return {
        "logreg": lambda: _lr(),
        "hgb": lambda: HistGradientBoostingClassifier(random_state=42, **HGB_PARAMS),
        "extratrees": lambda: SeedAvg(lambda sd: ExtraTreesClassifier(500, min_samples_leaf=3, max_features=0.5, n_jobs=-1, random_state=sd), SEEDS),
        "lgbm": lambda: SeedAvg(lambda sd: lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=25, subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5, verbose=-1, random_state=sd), SEEDS),
        "xgb": lambda: SeedAvg(lambda sd: xgb.XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.6, reg_lambda=5, min_child_weight=3, random_state=sd, verbosity=0), SEEDS),
        "catboost": lambda: CatBoostClassifier(iterations=600, learning_rate=0.04, depth=5, l2_leaf_reg=6, random_seed=0, verbose=0, thread_count=1, allow_writing_files=False),
    }


def main():
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    D = pos[0] if pos else HERE
    O = pos[1] if len(pos) > 1 else D
    os.makedirs(O, exist_ok=True)
    tr = pd.read_csv(os.path.join(D, "train.csv"))
    te = pd.read_csv(os.path.join(D, "test.csv"))
    print("Train:", tr.shape, " Test:", te.shape)
    y = tr["target"].map({c: i for i, c in enumerate(CLASSES)}).values
    X, gtr, str_ = prepare_features(tr)
    Xt, gte, ste = prepare_features(te)
    Xt = Xt.reindex(columns=X.columns, fill_value=0.0)
    imp = SimpleImputer(strategy="median").fit(X)
    Xp = pd.DataFrame(imp.transform(X), columns=X.columns)
    Xtp = pd.DataFrame(imp.transform(Xt), columns=X.columns)
    models = make_models()

    if "--cv" in sys.argv:
        oof = {k: np.zeros((len(Xp), 5)) for k in models}
        for a, b in GroupKFold(6).split(Xp, y, gtr):                     # hold out WHOLE cohort years
            Xa, Xb = add_learned_scores(Xp.iloc[a], y[a], gtr[a], str_[a], Xp.iloc[b], gtr[b], str_[b])
            for k, mk in models.items():
                oof[k][b] = mk().fit(Xa, y[a]).predict_proba(Xb)
        print("\nYEAR-GROUPED CV (macro-F1)")
        for k in oof:
            print(f" {k:<11s} {f1_score(y, oof[k].argmax(1), average='macro'):.4f}")
        P = np.mean(list(oof.values()), 0)
        p = P.argmax(1)
        print(f" {'ENSEMBLE':<11s} {f1_score(y, p, average='macro'):.4f}   accuracy={accuracy_score(y, p):.4f}")
        print(classification_report(y, p, target_names=CLASSES, digits=3))
        print(pd.DataFrame(confusion_matrix(y, p), index=[f"Actual {c}" for c in CLASSES], columns=[f"Pred {c}" for c in CLASSES]))

    # final fit on all labelled data
    Xa, Xb = add_learned_scores(Xp, y, gtr, str_, Xtp, gte, ste)
    print("Features:", Xa.shape[1])
    Pt = np.zeros((len(Xb), 5))
    for k, mk in models.items():
        Pt += mk().fit(Xa, y).predict_proba(Xb) / len(models)
    sub = pd.DataFrame({"id": te["id"], "target": [CLASSES[i] for i in Pt.argmax(1)]})
    assert len(sub) == len(te) and sub["id"].is_unique and sub["target"].isin(CLASSES).all()
    sub.to_csv(os.path.join(O, "submission.csv"), index=False)
    print(sub["target"].value_counts())
    print("Submission successfully exported to", os.path.join(O, "submission.csv"))


if __name__ == "__main__":
    main()
