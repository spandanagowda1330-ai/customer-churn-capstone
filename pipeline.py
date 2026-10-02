import json, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, roc_curve, precision_recall_curve, silhouette_score)
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", palette="deep")
RS = 42
R = {}  # results store

# ---------------------------------------------------------------- 1. DATA ACQUISITION
URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv("telco.csv")  # downloaded from URL above
R["raw_shape"] = list(df.shape)
R["dtypes"] = {k: str(v) for k, v in df.dtypes.items()}
R["duplicates"] = int(df.duplicated().sum())
R["dup_ids"] = int(df.customerID.duplicated().sum())

# ---------------------------------------------------------------- 2. CLEANING
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan))
R["missing_before"] = {k: int(v) for k, v in df.isna().sum().items() if v > 0}
R["tenure0_rows"] = int((df.tenure == 0).sum())
# those rows are brand-new customers (tenure 0) who have not been billed yet -> TotalCharges = 0
df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
df = df.drop(columns="customerID")
# harmonise "No internet service"/"No phone service" -> "No"
for c in ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]:
    df[c] = df[c].replace({"No internet service": "No", "No phone service": "No"})
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
R["clean_shape"] = list(df.shape)
R["churn_rate"] = float(df.Churn.mean())
R["churn_counts"] = {"No": int((df.Churn == 0).sum()), "Yes": int((df.Churn == 1).sum())}

# ---------------------------------------------------------------- 3. FEATURE ENGINEERING
svc = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df["NumAddOns"] = (df[svc] == "Yes").sum(axis=1)
df["AvgChargePerMonth"] = np.where(df.tenure > 0, df.TotalCharges / df.tenure.replace(0, np.nan), df.MonthlyCharges)
df["TenureGroup"] = pd.cut(df.tenure, [-1, 12, 24, 48, 72], labels=["0-12", "13-24", "25-48", "49-72"])
df["TenureGroup"] = df["TenureGroup"].astype(str)

# ---------------------------------------------------------------- 4. EDA
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
R["describe"] = df[num_cols].describe().round(2).to_dict()

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
cnt = df.Churn.map({0: "Stayed", 1: "Churned"}).value_counts()
ax[0].bar(cnt.index, cnt.values, color=["#4C72B0", "#C44E52"])
for i, v in enumerate(cnt.values): ax[0].text(i, v + 60, f"{v}\n({v/len(df):.1%})", ha="center")
ax[0].set_title("Customer churn distribution"); ax[0].set_ylim(0, 6200)
ax[1].pie(cnt.values, labels=cnt.index, autopct="%1.1f%%", colors=["#4C72B0", "#C44E52"], startangle=90)
ax[1].set_title("Churn share")
plt.tight_layout(); plt.savefig("fig01_churn_dist.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, c in zip(ax, num_cols):
    sns.histplot(data=df, x=c, hue=df.Churn.map({0: "Stayed", 1: "Churned"}), kde=True, bins=30, ax=a, element="step")
    a.set_title(f"{c} by churn")
plt.tight_layout(); plt.savefig("fig02_numeric_dist.png", dpi=150); plt.close()

cats = ["Contract", "InternetService", "PaymentMethod", "TechSupport", "PaperlessBilling", "SeniorCitizen"]
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
cat_rates = {}
for a, c in zip(ax.ravel(), cats):
    r = df.groupby(c).Churn.mean().sort_values(ascending=False) * 100
    cat_rates[c] = r.round(1).to_dict()
    sns.barplot(x=r.values, y=r.index, ax=a, color="#4C72B0")
    for i, v in enumerate(r.values): a.text(v + 0.5, i, f"{v:.1f}%", va="center")
    a.set_title(f"Churn rate by {c}"); a.set_xlabel("Churn rate (%)"); a.set_ylabel("")
    a.set_xlim(0, max(r.values) * 1.2)
plt.tight_layout(); plt.savefig("fig03_churn_by_cat.png", dpi=150); plt.close()
R["cat_rates"] = cat_rates

fig, ax = plt.subplots(figsize=(7, 5.5))
corr = df[num_cols + ["NumAddOns", "Churn"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Correlation heatmap (numeric features)")
plt.tight_layout(); plt.savefig("fig04_corr.png", dpi=150); plt.close()
R["corr_with_churn"] = corr["Churn"].round(3).to_dict()

fig, ax = plt.subplots(figsize=(7, 4.5))
sns.boxplot(data=df, x="Contract", y="MonthlyCharges", hue=df.Churn.map({0: "Stayed", 1: "Churned"}), ax=ax)
ax.set_title("Monthly charges by contract type and churn")
plt.tight_layout(); plt.savefig("fig05_box.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 5. SUPERVISED MODELLING
X = df.drop(columns="Churn"); y = df.Churn
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
R["split"] = {"train": len(Xtr), "test": len(Xte), "train_churn": float(ytr.mean()), "test_churn": float(yte.mean())}
num_f = ["tenure", "MonthlyCharges", "TotalCharges", "NumAddOns", "AvgChargePerMonth"]
cat_f = [c for c in X.columns if c not in num_f]
prep = ColumnTransformer([("num", StandardScaler(), num_f),
                          ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_f)])
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RS),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced", random_state=RS, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RS),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=500, random_state=RS),
}
skf = StratifiedKFold(5, shuffle=True, random_state=RS)
fitted, res, cv = {}, {}, {}
for name, m in models.items():
    pipe = Pipeline([("prep", prep), ("model", m)])
    cvs = cross_val_score(pipe, Xtr, ytr, cv=skf, scoring="roc_auc")
    cv[name] = (float(cvs.mean()), float(cvs.std()))
    pipe.fit(Xtr, ytr); fitted[name] = pipe
    p = pipe.predict_proba(Xte)[:, 1]; yh = (p >= 0.5).astype(int)
    res[name] = dict(Accuracy=accuracy_score(yte, yh), Precision=precision_score(yte, yh), Recall=recall_score(yte, yh),
                     F1=f1_score(yte, yh), ROC_AUC=roc_auc_score(yte, p), CV_AUC_mean=cv[name][0], CV_AUC_std=cv[name][1])

# hyper-parameter tuning on the best tree-based / linear candidates
gb = Pipeline([("prep", prep), ("model", GradientBoostingClassifier(random_state=RS))])
grid = {"model__n_estimators": [100, 200], "model__learning_rate": [0.03, 0.05, 0.1], "model__max_depth": [2, 3]}
gs = GridSearchCV(gb, grid, cv=skf, scoring="roc_auc", n_jobs=-1).fit(Xtr, ytr)
R["gb_best_params"] = gs.best_params_; R["gb_best_cv"] = float(gs.best_score_)
name = "Gradient Boosting (tuned)"; fitted[name] = gs.best_estimator_
p = gs.predict_proba(Xte)[:, 1]; yh = (p >= 0.5).astype(int)
res[name] = dict(Accuracy=accuracy_score(yte, yh), Precision=precision_score(yte, yh), Recall=recall_score(yte, yh),
                 F1=f1_score(yte, yh), ROC_AUC=roc_auc_score(yte, p), CV_AUC_mean=float(gs.best_score_), CV_AUC_std=float("nan"))

lr = Pipeline([("prep", prep), ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RS))])
gs2 = GridSearchCV(lr, {"model__C": [0.01, 0.1, 1, 10]}, cv=skf, scoring="roc_auc").fit(Xtr, ytr)
R["lr_best_params"] = gs2.best_params_
name = "Logistic Regression (tuned)"; fitted[name] = gs2.best_estimator_
p = gs2.predict_proba(Xte)[:, 1]; yh = (p >= 0.5).astype(int)
res[name] = dict(Accuracy=accuracy_score(yte, yh), Precision=precision_score(yte, yh), Recall=recall_score(yte, yh),
                 F1=f1_score(yte, yh), ROC_AUC=roc_auc_score(yte, p), CV_AUC_mean=float(gs2.best_score_), CV_AUC_std=float("nan"))
R["results"] = res

# ROC / PR curves
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for n, pipe in fitted.items():
    p = pipe.predict_proba(Xte)[:, 1]
    f, t, _ = roc_curve(yte, p); ax[0].plot(f, t, label=f"{n} ({roc_auc_score(yte, p):.3f})")
    pr, rc, _ = precision_recall_curve(yte, p); ax[1].plot(rc, pr, label=n)
ax[0].plot([0, 1], [0, 1], "k--"); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
ax[0].set_title("ROC curves (test set)"); ax[0].legend(fontsize=8)
ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-Recall curves (test set)")
plt.tight_layout(); plt.savefig("fig06_roc_pr.png", dpi=150); plt.close()

# best model selected by test ROC-AUC among tuned/untuned (selection by CV AUC to avoid test leakage)
best_name = max(res, key=lambda k: res[k]["CV_AUC_mean"]); R["best_model"] = best_name
best = fitted[best_name]; pb = best.predict_proba(Xte)[:, 1]

# threshold analysis
ths = np.arange(0.2, 0.71, 0.05); rows = []
for t in ths:
    yh = (pb >= t).astype(int)
    rows.append(dict(threshold=round(float(t), 2), precision=precision_score(yte, yh), recall=recall_score(yte, yh), f1=f1_score(yte, yh)))
R["thresholds"] = rows
bt = max(rows, key=lambda r: r["f1"]); R["best_threshold"] = bt
yh05 = (pb >= 0.5).astype(int); yhb = (pb >= bt["threshold"]).astype(int)
R["cm_05"] = confusion_matrix(yte, yh05).tolist(); R["cm_best"] = confusion_matrix(yte, yhb).tolist()
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
for a, cm, t in zip(ax, [R["cm_05"], R["cm_best"]], [0.5, bt["threshold"]]):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a, xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
    a.set_title(f"Confusion matrix (threshold = {t})"); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
plt.suptitle(best_name); plt.tight_layout(); plt.savefig("fig07_cm.png", dpi=150); plt.close()

# permutation importance (on original columns -> readable)
pi = permutation_importance(best, Xte, yte, scoring="roc_auc", n_repeats=10, random_state=RS, n_jobs=-1)
imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False)
R["perm_importance"] = imp.head(10).round(4).to_dict()
fig, ax = plt.subplots(figsize=(8, 5))
imp.head(12)[::-1].plot.barh(ax=ax, color="#55A868")
ax.set_title("Permutation importance (drop in ROC-AUC)"); ax.set_xlabel("Mean decrease in AUC")
plt.tight_layout(); plt.savefig("fig08_importance.png", dpi=150); plt.close()

# logistic regression coefficients for interpretability
lrp = fitted["Logistic Regression (tuned)"]
names = lrp.named_steps["prep"].get_feature_names_out()
coef = pd.Series(lrp.named_steps["model"].coef_[0], index=names).sort_values()
R["lr_coef_top_pos"] = coef.tail(6).round(3).to_dict(); R["lr_coef_top_neg"] = coef.head(6).round(3).to_dict()
sel = pd.concat([coef.head(8), coef.tail(8)])
fig, ax = plt.subplots(figsize=(8, 6))
sel.plot.barh(ax=ax, color=["#4C72B0" if v < 0 else "#C44E52" for v in sel])
ax.set_title("Logistic regression: strongest coefficients"); ax.set_xlabel("Coefficient (log-odds)")
plt.tight_layout(); plt.savefig("fig09_lr_coef.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 6. UNSUPERVISED: K-MEANS + PCA
Xc = df.drop(columns=["Churn"])
Xs = prep.fit_transform(Xc)
inertia, sil = [], []
ks = range(2, 9)
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(Xs)
    inertia.append(km.inertia_); sil.append(silhouette_score(Xs, km.labels_, sample_size=3000, random_state=RS))
R["k_sil"] = dict(zip([int(k) for k in ks], [round(s, 4) for s in sil]))
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(list(ks), inertia, "o-"); ax[0].set_title("Elbow method"); ax[0].set_xlabel("k"); ax[0].set_ylabel("Inertia")
ax[1].plot(list(ks), sil, "o-", color="#C44E52"); ax[1].set_title("Silhouette score"); ax[1].set_xlabel("k"); ax[1].set_ylabel("Score")
plt.tight_layout(); plt.savefig("fig10_kmeans_select.png", dpi=150); plt.close()

K = 4
km = KMeans(n_clusters=K, n_init=10, random_state=RS).fit(Xs)
df["Cluster"] = km.labels_
R["k_used"] = K; R["sil_k"] = float(silhouette_score(Xs, km.labels_, sample_size=3000, random_state=RS))
pca = PCA(n_components=2, random_state=RS); Z = pca.fit_transform(Xs)
R["pca_var"] = [float(v) for v in pca.explained_variance_ratio_]
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
sc = ax[0].scatter(Z[:, 0], Z[:, 1], c=df.Cluster, cmap="tab10", s=8, alpha=.6)
ax[0].set_title("K-Means clusters in PCA space"); ax[0].set_xlabel("PC1"); ax[0].set_ylabel("PC2")
ax[0].legend(*sc.legend_elements(), title="Cluster")
ax[1].scatter(Z[:, 0], Z[:, 1], c=df.Churn.map({0: "#4C72B0", 1: "#C44E52"}), s=8, alpha=.5)
ax[1].set_title("Same PCA space coloured by churn (red = churned)"); ax[1].set_xlabel("PC1"); ax[1].set_ylabel("PC2")
plt.tight_layout(); plt.savefig("fig11_pca_clusters.png", dpi=150); plt.close()

prof = df.groupby("Cluster").agg(Customers=("Churn", "size"), ChurnRate=("Churn", "mean"), Tenure=("tenure", "mean"),
                                 MonthlyCharges=("MonthlyCharges", "mean"), NumAddOns=("NumAddOns", "mean"),
                                 MonthToMonth=("Contract", lambda s: (s == "Month-to-month").mean()),
                                 Fiber=("InternetService", lambda s: (s == "Fiber optic").mean()),
                                 NoInternet=("InternetService", lambda s: (s == "No").mean())).round(3)
R["cluster_profile"] = prof.reset_index().to_dict(orient="records")
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(x=prof.index, y=prof.ChurnRate * 100, ax=ax, color="#4C72B0")
for i, v in enumerate(prof.ChurnRate * 100): ax.text(i, v + .5, f"{v:.1f}%", ha="center")
ax.axhline(R["churn_rate"] * 100, color="red", ls="--", label="Overall churn"); ax.legend()
ax.set_title("Churn rate per customer segment"); ax.set_xlabel("Cluster"); ax.set_ylabel("Churn rate (%)")
plt.tight_layout(); plt.savefig("fig12_cluster_churn.png", dpi=150); plt.close()

# ---------------------------------------------------------------- 7. BUSINESS VIEW: top-decile lift
tmp = pd.DataFrame({"y": yte.values, "p": pb}).sort_values("p", ascending=False)
n10 = int(len(tmp) * 0.2)
R["top20_capture"] = float(tmp.y.iloc[:n10].sum() / tmp.y.sum()); R["top20_rate"] = float(tmp.y.iloc[:n10].mean())

json.dump(R, open("results.json", "w"), indent=1, default=float)
print(json.dumps({k: R[k] for k in ["raw_shape", "missing_before", "churn_rate", "best_model", "best_threshold", "top20_capture", "k_sil", "sil_k", "pca_var"]}, indent=1, default=float))
print(pd.DataFrame(res).T.round(4))
print(prof)
print(R["perm_importance"]); print(R["cm_05"], R["cm_best"]); print(cat_rates); print(R["gb_best_params"], R["lr_best_params"])
print(R["corr_with_churn"]); print(R["lr_coef_top_pos"], R["lr_coef_top_neg"])
