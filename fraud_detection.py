# =============================================================================
# Mid-Term Practical Assignment
# From Classical Data Analysis to Neural Networks for Fraud Detection
# Python for Business Economics and Finance
# =============================================================================
# INSTRUCTIONS: Run this script from the project root directory.
# Required libraries: pandas, numpy, matplotlib, seaborn, scikit-learn
# Dataset: synthetic_transactions.csv (placed in the same directory or path below)
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    mean_squared_error, r2_score
)

# Output directory for figures
FIG_DIR = "figs"
os.makedirs(FIG_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = {0: "#4C8BF5", 1: "#E84855"}


# =============================================================================
# PART A – DATA LOADING AND PREPROCESSING
# =============================================================================

print("=" * 60)
print("PART A – Data Loading and Preprocessing")
print("=" * 60)

# A.a – Load CSV
df = pd.read_csv("/mnt/user-data/uploads/synthetic_transactions.csv")

# A.b – Shape and first rows
print(f"\nDimensions: {df.shape[0]:,} rows × {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

# A.c – Variable types
print("\nVariable dtypes:")
print(df.dtypes.to_string())

# A.d – Missing values
missing = df.isnull().sum()
print("\nMissing values per column:")
print(missing[missing > 0])
# Strategy: impute 'days_since_last_txn' with median (robust to outliers)
# This column is NaN for new customers who have no prior transaction.
df['days_since_last_txn'] = df['days_since_last_txn'].fillna(
    df['days_since_last_txn'].median()
)
print(">> 'days_since_last_txn' imputed with median.")

# A.e – Categorical vs numerical variables
CAT_COLS = ['merchant_type', 'channel', 'region']
NUM_COLS = [c for c in df.columns
            if df[c].dtype != object
            and c not in ['transaction_id', 'customer_id', 'fraud', 'loss_amount']]
print(f"\nCategorical columns ({len(CAT_COLS)}): {CAT_COLS}")
print(f"Numerical columns  ({len(NUM_COLS)}): {NUM_COLS}")

# A.f – Preprocessing: encode, scale, split
# Drop ID columns and loss_amount (target leakage for fraud model)
DROP_COLS = ['transaction_id', 'customer_id', 'loss_amount']
df_enc = pd.get_dummies(df.drop(columns=DROP_COLS), columns=CAT_COLS, drop_first=True)

FEATURE_COLS = [c for c in df_enc.columns if c != 'fraud']
X = df_enc[FEATURE_COLS].fillna(df_enc[FEATURE_COLS].median())  # safety impute
y = df_enc['fraud']

# Scale: required for logistic regression and neural networks
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Stratified split to preserve fraud class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain set: {X_train.shape[0]:,} samples  |  Test set: {X_test.shape[0]:,} samples")


# =============================================================================
# PART B – DESCRIPTIVE STATISTICS AND VISUALIZATION
# =============================================================================

print("\n" + "=" * 60)
print("PART B – Descriptive Statistics and Visualization")
print("=" * 60)

# B.a – Summary statistics
print("\nSummary statistics (selected numerical variables):")
print(df[['amount', 'risk_score', 'amount_ratio', 'customer_age', 'days_since_last_txn']
         ].describe().round(2).to_string())

# B.b – Fraud proportion
fraud_rate = df['fraud'].mean()
print(f"\nFraud rate: {fraud_rate:.4f}  ({fraud_rate*100:.2f}%)")
print(f"  Non-fraud: {(y == 0).sum():,}  |  Fraud: {(y == 1).sum():,}")

# B.c – Visualizations

# Figure 1: Fraud class bar chart
fig, ax = plt.subplots(figsize=(5, 3.5))
counts = df['fraud'].value_counts().sort_index()
ax.bar(['Non-Fraud', 'Fraud'], counts.values,
       color=[PALETTE[0], PALETTE[1]], edgecolor='white')
for i, v in enumerate(counts.values):
    ax.text(i, v + 50, f'{v:,}', ha='center', fontweight='bold')
ax.set_title('Class Distribution', fontweight='bold')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_fraud_bar.png", dpi=140, bbox_inches='tight')
plt.close()
print("\n>> Figure 1 saved: class distribution bar chart")

# Figure 2: Amount distribution by fraud (log scale)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, log in zip(axes, [False, True]):
    for label, grp in df.groupby('fraud'):
        data = np.log1p(grp['amount']) if log else grp['amount']
        ax.hist(data, bins=50, alpha=0.6,
                label=f"{'Fraud' if label else 'Non-Fraud'}",
                color=PALETTE[label], edgecolor='none')
    ax.set_title(f"Amount {'(log scale)' if log else '(raw)'}")
    ax.set_xlabel("log(1 + amount)" if log else "Amount")
    ax.legend(fontsize=9)
axes[0].set_ylabel("Frequency")
plt.suptitle("Transaction Amount Distribution by Fraud Status", fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_amount_hist.png", dpi=140, bbox_inches='tight')
plt.close()
print(">> Figure 2 saved: amount histograms")

# Figure 3: Boxplot amount by fraud
fig, ax = plt.subplots(figsize=(6, 4))
df.boxplot(column='amount', by='fraud', ax=ax,
           medianprops=dict(color='red', linewidth=2),
           flierprops=dict(marker='.', markersize=2, alpha=0.3))
ax.set_title('Transaction Amount by Fraud Label', fontweight='bold')
ax.set_xlabel('Fraud (0 = Non-Fraud, 1 = Fraud)')
ax.set_ylabel('Amount')
plt.suptitle("")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig3_boxplot.png", dpi=140, bbox_inches='tight')
plt.close()
print(">> Figure 3 saved: boxplot amount by fraud")

# Figure 4: Fraud rate by merchant type
fig, ax = plt.subplots(figsize=(8, 4))
merch_fraud = df.groupby('merchant_type')['fraud'].mean().sort_values(ascending=False)
merch_fraud.plot(kind='bar', ax=ax, color='#E84855', edgecolor='white')
ax.set_title('Fraud Rate by Merchant Type', fontweight='bold')
ax.set_xlabel('Merchant Type')
ax.set_ylabel('Fraud Rate')
ax.tick_params(axis='x', rotation=30)
ax.yaxis.set_major_formatter(
    matplotlib.ticker.PercentFormatter(xmax=1, decimals=1))
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig4_merchant.png", dpi=140, bbox_inches='tight')
plt.close()
print(">> Figure 4 saved: fraud rate by merchant type")

# Figure 5: Correlation heatmap
fig, ax = plt.subplots(figsize=(11, 8))
corr_cols = ['amount', 'risk_score', 'amount_ratio', 'customer_age',
             'income_band', 'tenure_months', 'daily_txn_count',
             'days_since_last_txn', 'is_foreign', 'high_risk_merchant',
             'night_txn', 'weekend', 'card_present', 'fraud']
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, annot_kws={'size': 8})
ax.set_title('Correlation Heatmap', fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig5_corr.png", dpi=140, bbox_inches='tight')
plt.close()
print(">> Figure 5 saved: correlation heatmap")

# B.d – Descriptive comparison table (fraud vs non-fraud)
COMP_VARS = ['amount', 'risk_score', 'amount_ratio', 'days_since_last_txn',
             'daily_txn_count', 'is_foreign', 'high_risk_merchant', 'night_txn']
comp_table = df.groupby('fraud')[COMP_VARS].mean().T
comp_table.columns = ['Non-Fraud', 'Fraud']
print("\nDescriptive comparison (means by fraud label):")
print(comp_table.round(4).to_string())


# =============================================================================
# PART C – LINEAR REGRESSION (predicting transaction amount)
# =============================================================================

print("\n" + "=" * 60)
print("PART C – Linear Regression")
print("=" * 60)

# Predictors chosen for their theoretical link to transaction amount:
# avg_prev_amount: prior spending level; income_band: purchasing power;
# amount_ratio: relative size of transaction; risk_score: risk profile;
# is_foreign, high_risk_merchant: contextual flags.
LIN_FEATURES = ['avg_prev_amount', 'income_band', 'customer_age',
                'tenure_months', 'daily_txn_count', 'risk_score',
                'is_foreign', 'high_risk_merchant', 'amount_ratio']

X_lin = df[LIN_FEATURES].fillna(df[LIN_FEATURES].median())
y_lin = df['amount']

Xl_train, Xl_test, yl_train, yl_test = train_test_split(
    X_lin, y_lin, test_size=0.2, random_state=42
)
sc_lin = StandardScaler()
Xl_train_s = sc_lin.fit_transform(Xl_train)
Xl_test_s  = sc_lin.transform(Xl_test)

lin_model = LinearRegression()
lin_model.fit(Xl_train_s, yl_train)
yl_pred = lin_model.predict(Xl_test_s)

rmse_lin = np.sqrt(mean_squared_error(yl_test, yl_pred))
r2_lin   = r2_score(yl_test, yl_pred)
print(f"\nLinear Regression — RMSE: {rmse_lin:.2f}  |  R²: {r2_lin:.3f}")

# Coefficient table
coef_df = pd.DataFrame({'Feature': LIN_FEATURES, 'Coef': lin_model.coef_}
                       ).sort_values('Coef', key=abs, ascending=False)
print("\nCoefficients (standardised features, sorted by |coef|):")
print(coef_df.round(4).to_string(index=False))

# Figure 6: actual vs predicted + coefficients
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax.scatter(yl_test, yl_pred, alpha=0.3, s=8, color='#4C8BF5')
lims = [min(yl_test.min(), yl_pred.min()), max(yl_test.max(), yl_pred.max())]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
ax.set_xlabel("Actual Amount"); ax.set_ylabel("Predicted Amount")
ax.set_title("Linear Reg: Actual vs Predicted", fontweight='bold')
ax.legend()
ax = axes[1]
colors_bar = ['#E84855' if c > 0 else '#4C8BF5' for c in coef_df['Coef']]
ax.barh(coef_df['Feature'], coef_df['Coef'], color=colors_bar, edgecolor='white')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title("Regression Coefficients", fontweight='bold')
ax.set_xlabel("Coefficient (standardised)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig6_linreg.png", dpi=140, bbox_inches='tight')
plt.close()
print(">> Figure 6 saved: linear regression plots")

# Interpretation of top 3 coefficients:
# 1. avg_prev_amount (+): strongest predictor — past spending predicts current spending.
# 2. amount_ratio (+): large transactions relative to history → higher absolute amount.
# 3. income_band (+): higher income → higher spending capacity.
# Linear regression is inappropriate for fraud (binary target) because it produces
# unbounded predictions and assumes additive linearity — use logistic regression instead.


# =============================================================================
# PART D – LOGISTIC REGRESSION FOR FRAUD CLASSIFICATION
# =============================================================================

print("\n" + "=" * 60)
print("PART D – Logistic Regression")
print("=" * 60)

# class_weight='balanced' adjusts for the 2.13% fraud rate:
# it effectively upweights fraud samples during training.
log_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)
y_prob_log = log_model.predict_proba(X_test)[:, 1]

cm_log   = confusion_matrix(y_test, y_pred_log)
prec_log = precision_score(y_test, y_pred_log)
rec_log  = recall_score(y_test, y_pred_log)
f1_log   = f1_score(y_test, y_pred_log)
auc_log  = roc_auc_score(y_test, y_prob_log)

print(f"\nConfusion Matrix:\n{cm_log}")
print(f"\nPrecision : {prec_log:.4f}")
print(f"Recall    : {rec_log:.4f}")
print(f"F1-Score  : {f1_log:.4f}")
print(f"ROC-AUC   : {auc_log:.4f}")

# Note: With imbalanced data (97.87% non-fraud), a naive classifier predicting all
# non-fraud achieves 97.87% accuracy but 0% recall on fraud.
# Precision, Recall, F1 and AUC measure minority-class performance properly.

# Top logistic predictors
log_coef_df = pd.DataFrame({'Feature': FEATURE_COLS, 'Coef': log_model.coef_[0]}
                           ).sort_values('Coef', key=abs, ascending=False)
print("\nTop 10 predictors by logistic coefficient magnitude:")
print(log_coef_df.head(10).round(4).to_string(index=False))

# Figure 7: confusion matrix + ROC
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Pred Non-Fraud', 'Pred Fraud'],
            yticklabels=['True Non-Fraud', 'True Fraud'])
ax.set_title('Logistic Regression — Confusion Matrix', fontweight='bold')
ax = axes[1]
fpr_l, tpr_l, _ = roc_curve(y_test, y_prob_log)
ax.plot(fpr_l, tpr_l, color='#4C8BF5', lw=2, label=f'Logistic (AUC={auc_log:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title('ROC Curve', fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig7_logistic.png", dpi=140, bbox_inches='tight')
plt.close()
print(">> Figure 7 saved: logistic regression plots")


# =============================================================================
# PART E – PROGRAMMING WITH CLASSES
# =============================================================================

print("\n" + "=" * 60)
print("PART E – FraudAnalysisPipeline Class")
print("=" * 60)


class FraudAnalysisPipeline:
    """
    Encapsulates the full fraud detection modelling workflow.

    Attributes
    ----------
    df_raw       : pd.DataFrame   – raw input data
    scaler       : StandardScaler – fitted feature scaler
    log_model    : LogisticRegression
    nn_model     : MLPClassifier
    feature_cols : list[str]      – feature names after encoding
    results      : dict           – stores evaluation metrics per model

    Methods
    -------
    preprocess()          → self   (enables method chaining)
    descriptive_stats()   → pd.DataFrame
    fit_logistic_model()  → self
    fit_nn_model()        → self
    evaluate_model(name)  → dict
    """

    def __init__(self, df: pd.DataFrame):
        self.df_raw    = df.copy()
        self.scaler    = StandardScaler()
        self.log_model = None
        self.nn_model  = None
        self.results   = {}
        self.X_train   = self.X_test  = None
        self.y_train   = self.y_test  = None
        self.feature_cols = []

    def preprocess(self, cat_cols=None, drop_cols=None):
        """
        One-hot encode categoricals, impute missing values with median,
        standardise features, and create a stratified 80/20 train/test split.
        Returns self for method chaining.
        """
        cat_cols  = cat_cols  or ['merchant_type', 'channel', 'region']
        drop_cols = drop_cols or ['transaction_id', 'customer_id', 'loss_amount']

        df = self.df_raw.copy()
        # Median imputation for numerical NaNs
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].median())

        df_enc = pd.get_dummies(df.drop(columns=drop_cols),
                                columns=cat_cols, drop_first=True)
        self.feature_cols = [c for c in df_enc.columns if c != 'fraud']

        X = df_enc[self.feature_cols].values
        y = df_enc['fraud'].values
        X_s = self.scaler.fit_transform(X)

        (self.X_train, self.X_test,
         self.y_train, self.y_test) = train_test_split(
            X_s, y, test_size=0.2, random_state=42, stratify=y)
        return self

    def descriptive_stats(self):
        """Return mean feature values grouped by fraud label."""
        cols = ['amount', 'risk_score', 'amount_ratio',
                'is_foreign', 'high_risk_merchant', 'night_txn']
        return self.df_raw.groupby('fraud')[cols].mean().round(4)

    def fit_logistic_model(self):
        """
        Fit balanced logistic regression. class_weight='balanced' compensates
        for the imbalanced target without resampling the data.
        Returns self for method chaining.
        """
        self.log_model = LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42)
        self.log_model.fit(self.X_train, self.y_train)
        return self

    def fit_nn_model(self):
        """
        Fit a two-hidden-layer MLP (64→32 neurons, ReLU, Adam).
        Early stopping prevents overfitting.
        Returns self for method chaining.
        """
        self.nn_model = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15
        )
        self.nn_model.fit(self.X_train, self.y_train)
        return self

    def evaluate_model(self, model_name: str = 'logistic') -> dict:
        """
        Compute precision, recall, F1, and ROC-AUC for a fitted model.

        Parameters
        ----------
        model_name : 'logistic' or 'nn'

        Returns
        -------
        dict with keys: precision, recall, f1, auc
        """
        model = self.log_model if model_name == 'logistic' else self.nn_model
        if model is None:
            raise ValueError(f"Model '{model_name}' has not been fitted yet.")
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]
        self.results[model_name] = {
            'precision': precision_score(self.y_test, y_pred),
            'recall':    recall_score(self.y_test, y_pred),
            'f1':        f1_score(self.y_test, y_pred),
            'auc':       roc_auc_score(self.y_test, y_prob),
        }
        return self.results[model_name]


# Demonstrate the class in action
pipeline = FraudAnalysisPipeline(df)
pipeline.preprocess().fit_logistic_model().fit_nn_model()

res_log = pipeline.evaluate_model('logistic')
res_nn  = pipeline.evaluate_model('nn')

print("\nPipeline class results:")
print(f"  Logistic  → F1={res_log['f1']:.4f}  AUC={res_log['auc']:.4f}")
print(f"  NeuralNet → F1={res_nn['f1']:.4f}  AUC={res_nn['auc']:.4f}")
print("\nDescriptive stats from class:")
print(pipeline.descriptive_stats().to_string())


# =============================================================================
# PART F – NEURAL NETWORK CLASSIFIER
# =============================================================================

print("\n" + "=" * 60)
print("PART F – Neural Network Classifier")
print("=" * 60)

# Architecture: input → 64 (ReLU) → 32 (ReLU) → 1 (sigmoid)
# Rationale:
#   - Two hidden layers capture non-linear feature interactions
#   - ReLU avoids vanishing gradients; sigmoid output gives fraud probability
#   - Adam optimiser: adaptive learning rates, fast convergence
#   - Early stopping (patience=15): prevents overfitting on training set
#   - Loss: binary cross-entropy (standard for binary classification)

nn_model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15
)
nn_model.fit(X_train, y_train)

y_pred_nn = nn_model.predict(X_test)
y_prob_nn = nn_model.predict_proba(X_test)[:, 1]

cm_nn   = confusion_matrix(y_test, y_pred_nn)
prec_nn = precision_score(y_test, y_pred_nn)
rec_nn  = recall_score(y_test, y_pred_nn)
f1_nn   = f1_score(y_test, y_pred_nn)
auc_nn  = roc_auc_score(y_test, y_prob_nn)

print(f"\nConfusion Matrix:\n{cm_nn}")
print(f"\nPrecision : {prec_nn:.4f}")
print(f"Recall    : {rec_nn:.4f}")
print(f"F1-Score  : {f1_nn:.4f}")
print(f"ROC-AUC   : {auc_nn:.4f}")

# Figure 8: confusion matrices + ROC comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
# Neural net confusion matrix
sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Oranges', ax=axes[0],
            xticklabels=['Pred NF', 'Pred F'],
            yticklabels=['True NF', 'True F'])
axes[0].set_title('Neural Network\nConfusion Matrix', fontweight='bold')
# Logistic confusion matrix for comparison
sns.heatmap(cm_log, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=['Pred NF', 'Pred F'],
            yticklabels=['True NF', 'True F'])
axes[1].set_title('Logistic Regression\nConfusion Matrix', fontweight='bold')
# ROC curves
ax = axes[2]
fpr_n, tpr_n, _ = roc_curve(y_test, y_prob_nn)
ax.plot(fpr_l, tpr_l, color='#4C8BF5', lw=2, label=f'Logistic (AUC={auc_log:.3f})')
ax.plot(fpr_n, tpr_n, color='#E84855', lw=2, label=f'Neural Net (AUC={auc_nn:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title('ROC Curve Comparison', fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig8_nn.png", dpi=140, bbox_inches='tight')
plt.close()
print(">> Figure 8 saved: neural network plots")


# =============================================================================
# PART G – AUTOENCODER FOR ANOMALY DETECTION
# =============================================================================

print("\n" + "=" * 60)
print("PART G – Autoencoder Anomaly Detection")
print("=" * 60)

# Logic: Train autoencoder ONLY on non-fraud transactions.
# The network learns to efficiently reconstruct normal behaviour.
# A fraudulent transaction deviates from normal patterns →
# reconstruction error (MSE) is higher → anomaly flag.

# G.a – Prepare data: train only on non-fraud examples
train_nf_mask = y_train == 0
X_ae_train = X_train[train_nf_mask]
print(f"\nAutoencoder training on {X_ae_train.shape[0]:,} non-fraud samples.")

# G.b – Build autoencoder (symmetric bottleneck MLP via MLPRegressor)
# Architecture: input → 32 → 8 (bottleneck) → 32 → output (same shape as input)
# Using MLPRegressor to reconstruct the input vector (self-supervised learning)
autoencoder = MLPRegressor(
    hidden_layer_sizes=(32, 8, 32),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15
)
autoencoder.fit(X_ae_train, X_ae_train)   # target = input (reconstruction)
print("Autoencoder training complete.")

# G.c – Reconstruction errors on test set
X_test_recon = autoencoder.predict(X_test)
recon_errors = np.mean((X_test - X_test_recon) ** 2, axis=1)

# G.d – Threshold selection: 95th percentile of non-fraud reconstruction errors.
# This controls the false positive rate: ~5% of non-fraud will be flagged.
# The choice is a business trade-off: lower threshold → higher recall, more false alarms.
nf_errors = recon_errors[y_test == 0]
threshold  = np.percentile(nf_errors, 95)
print(f"\nDecision threshold (95th pct of non-fraud errors): {threshold:.4f}")

# G.e – Classify as anomaly if reconstruction error > threshold
y_pred_ae = (recon_errors > threshold).astype(int)

# G.f – Evaluate against observed fraud labels
cm_ae   = confusion_matrix(y_test, y_pred_ae)
prec_ae = precision_score(y_test, y_pred_ae, zero_division=0)
rec_ae  = recall_score(y_test, y_pred_ae, zero_division=0)
f1_ae   = f1_score(y_test, y_pred_ae, zero_division=0)
auc_ae  = roc_auc_score(y_test, recon_errors)   # AUC uses raw error as score

print(f"\nConfusion Matrix:\n{cm_ae}")
print(f"Precision : {prec_ae:.4f}")
print(f"Recall    : {rec_ae:.4f}")
print(f"F1-Score  : {f1_ae:.4f}")
print(f"ROC-AUC   : {auc_ae:.4f}")

# Figure 9: reconstruction error distributions + confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
bins = np.linspace(0, np.percentile(recon_errors, 99), 80)
ax.hist(recon_errors[y_test == 0], bins=bins, alpha=0.6,
        color=PALETTE[0], label='Non-Fraud')
ax.hist(recon_errors[y_test == 1], bins=bins, alpha=0.6,
        color=PALETTE[1], label='Fraud')
ax.axvline(threshold, color='black', linestyle='--', linewidth=1.5,
           label=f'Threshold={threshold:.3f}')
ax.set_xlabel("Reconstruction Error (MSE)")
ax.set_ylabel("Frequency")
ax.set_title("Autoencoder: Reconstruction Errors", fontweight='bold')
ax.legend(fontsize=9)
ax = axes[1]
sns.heatmap(cm_ae, annot=True, fmt='d', cmap='Greens', ax=ax,
            xticklabels=['Pred Normal', 'Pred Anomaly'],
            yticklabels=['True Non-Fraud', 'True Fraud'])
ax.set_title("Autoencoder — Confusion Matrix", fontweight='bold')
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig9_autoencoder.png", dpi=140, bbox_inches='tight')
plt.close()
print(">> Figure 9 saved: autoencoder plots")


# =============================================================================
# PART H – FINAL COMPARISON AND REFLECTION
# =============================================================================

print("\n" + "=" * 60)
print("PART H – Final Comparison and Reflection")
print("=" * 60)

print("\n--- Linear Regression (predict amount) ---")
print(f"  RMSE = {rmse_lin:.2f},  R² = {r2_lin:.3f}")
print("  Useful for explaining transaction magnitude, not fraud.")

print("\n--- Classification model comparison (test set) ---")
comparison = pd.DataFrame({
    'Model':     ['Logistic Regression', 'Neural Network', 'Autoencoder (unsup.)'],
    'Precision': [prec_log, prec_nn, prec_ae],
    'Recall':    [rec_log,  rec_nn,  rec_ae],
    'F1-Score':  [f1_log,   f1_nn,   f1_ae],
    'ROC-AUC':   [auc_log,  auc_nn,  auc_ae],
})
print(comparison.round(4).to_string(index=False))

print("""
Recommendation:
  1. Logistic Regression – primary real-time scorer (interpretable, fast, auditable).
  2. Neural Network      – secondary layer for complex non-linear patterns.
  3. Autoencoder         – unsupervised anomaly sentinel for novel/unknown fraud types.

Limitations:
  - Synthetic data leads to overly optimistic supervised metrics.
  - No hyperparameter tuning was applied (cross-validated grid search recommended).
  - The autoencoder uses sklearn MLP; Keras/PyTorch would offer more flexibility.
  - SMOTE or other oversampling strategies could further improve minority-class recall.
""")

print("=" * 60)
print("All figures saved to:", FIG_DIR)
print("Script completed successfully.")
print("=" * 60)
