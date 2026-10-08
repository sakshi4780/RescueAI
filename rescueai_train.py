"""
RescueAI: simulate sessions, train the abandonment + driver models, export weights.

Run:  pip install numpy pandas scikit-learn
      python rescueai_train.py

Outputs: rescueai_sessions.csv, rescueai_weights.json, feature_importance.csv

HONESTY NOTE: the data is SIMULATED. Labels (including the "driver") come from
rules written in simulate() below, so high scores mostly show the pipeline works,
not that real shoppers behave this way. In production, abandonment is observed
directly, while the true driver is not: you would infer it from exit surveys and
A/B-tested interventions.
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report, confusion_matrix)

SEED = 42
FEATURES = ["price", "tprod", "ship", "pcmp", "cart", "tco", "delivery_days", "checkout"]
DRIVERS = ["none", "delivery", "price", "friction"]


def simulate(n=20000, seed=SEED):
    rng = np.random.default_rng(seed)
    driver = rng.choice(DRIVERS, size=n, p=[0.40, 0.20, 0.25, 0.15])
    is_d, is_p, is_f = (driver == "delivery"), (driver == "price"), (driver == "friction")

    price = np.clip(np.round(rng.lognormal(7.8, 0.5, n) / 100) * 100 - 1, 499, 9999)
    tprod = np.clip(rng.gamma(2, 1.2, n), 0, 10)
    ship = np.clip(rng.poisson(0.7 + 2.5 * is_d), 0, 8)
    pcmp = np.clip(rng.poisson(0.5 + 4.0 * is_p + 0.0004 * (price - 2500) * is_p), 0, 10)
    cart = np.clip(1 + rng.poisson(0.3, n), 1, 4)
    tco = np.clip(rng.gamma(2, 0.4 + 0.5 * is_f + 0.2 * is_d), 0, 6)
    delivery_days = np.clip(rng.integers(1, 4, n) + is_d * rng.integers(1, 4, n), 1, 7)
    checkout = rng.binomial(1, 0.85, n)

    # Abandonment: driven mostly by the hidden driver, plus noise and mild signal effects.
    base = np.select([driver == "none", is_d, is_p, is_f], [0.08, 0.78, 0.75, 0.70])
    p = np.clip(base + 0.015 * (tco - 1) + 0.00002 * (price - 2500) - 0.01 * (tprod - 2), 0.02, 0.97)
    abandoned = rng.binomial(1, p)

    df = pd.DataFrame(dict(price=price, tprod=tprod, ship=ship, pcmp=pcmp, cart=cart,
                           tco=tco, delivery_days=delivery_days, checkout=checkout,
                           abandoned=abandoned, driver=driver))
    df.loc[df.abandoned == 0, "driver"] = "none"  # drivers only matter if they left
    return df


def main():
    df = simulate()
    df.to_csv("rescueai_sessions.csv", index=False)
    print(f"Simulated {len(df):,} sessions | abandonment rate {df.abandoned.mean():.1%}")

    # ---- Model 1: will this session be abandoned? ----
    X, y = df[FEATURES], df["abandoned"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    rf.fit(Xtr, ytr)
    pr = rf.predict_proba(Xte)[:, 1]
    print(f"\n[Abandonment] Random Forest  ROC-AUC {roc_auc_score(yte, pr):.3f} | "
          f"PR-AUC {average_precision_score(yte, pr):.3f} (baseline PR-AUC {yte.mean():.3f})")

    # Interpretable twin: logistic regression, whose weights map directly to the web demo.
    Xl = Xtr.copy(); Xl["price"] = (Xl["price"] - 2500) / 1000   # per ₹1,000 above ₹2,500
    lr = LogisticRegression(max_iter=3000).fit(Xl, ytr)
    Xt = Xte.copy(); Xt["price"] = (Xt["price"] - 2500) / 1000
    print(f"[Abandonment] Logistic Reg.  ROC-AUC {roc_auc_score(yte, lr.predict_proba(Xt)[:, 1]):.3f}")

    fi = pd.Series(rf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    fi.to_csv("feature_importance.csv", header=["importance"])
    print("\nFeature importance (RF):\n" + fi.round(3).to_string())

    # ---- Model 2: given abandonment, which driver is most likely? ----
    ab = df[df.abandoned == 1]
    Xa, ya = ab[FEATURES], ab["driver"]
    Xatr, Xate, yatr, yate = train_test_split(Xa, ya, test_size=0.25, stratify=ya, random_state=SEED)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    clf.fit(Xatr, yatr)
    pred = clf.predict(Xate)
    print("\n[Driver model] trained on abandoned sessions only")
    print(classification_report(yate, pred, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred, order", list(clf.classes_), ")")
    print(confusion_matrix(yate, pred, labels=clf.classes_))

    # ---- Export logistic weights for the demo ----
    out = {"intercept": float(lr.intercept_[0]),
           "weights": {f: float(c) for f, c in zip(FEATURES, lr.coef_[0])},
           "note": "price weight is per ₹1,000 above ₹2,500; divide by 1000 for per-₹"}
    with open("rescueai_weights.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved rescueai_sessions.csv, feature_importance.csv, rescueai_weights.json")

    # ---- Decision layer: what the product DOES with the prediction ----
    # Next step for a real product: A/B test each intervention and measure uplift,
    # not just prediction accuracy.


if __name__ == "__main__":
    main()
