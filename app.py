from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AML Alert Escalation — ML Analysis",
    page_icon="🔍",
    layout="wide",
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
EDA_DIR = BASE_DIR / "eda_data"
DATA_DIR = BASE_DIR / "data"


# ============================================================
# SAFE LOADERS
# ============================================================

@st.cache_data
def load_csv(path):
    """
    Load a CSV without crashing the Streamlit app if the file is
    missing, empty, or malformed.
    """
    path = Path(path)

    if not path.exists() or path.stat().st_size == 0:
        return None

    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError):
        return None


target_distribution = load_csv(EDA_DIR / "target_distribution.csv")
transaction_direction = load_csv(EDA_DIR / "transaction_direction.csv")
transaction_types = load_csv(EDA_DIR / "transaction_types.csv")

# Optional files. The app does NOT depend on them.
alert_behavior = load_csv(EDA_DIR / "alert_behavior.csv")
feature_importance = load_csv(EDA_DIR / "feature_importance.csv")
model_comparison = load_csv(EDA_DIR / "model_comparison.csv")

# Small alert-level CSVs are safe to load if they exist.
test_signals = load_csv(DATA_DIR / "test_signals.csv")


# ============================================================
# HELPERS
# ============================================================

def numeric_series(df, column):
    return pd.to_numeric(df[column], errors="coerce")


def total_count(df):
    if df is None or "count" not in df.columns:
        return None
    return int(numeric_series(df, "count").fillna(0).sum())


def pct(value, total):
    if not total:
        return 0.0
    return value / total * 100


def available_columns(df, columns):
    if df is None:
        return []
    return [column for column in columns if column in df.columns]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

pages = [
    "1. Overview",
    "2. Dataset & Structure",
    "3. Target Distribution",
    "4. Transaction Behavior",
    "5. Key Behavioral Patterns",
    "6. EDA → Feature Engineering",
    "7. How the Model Works",
    "8. Model Features",
    "9. Validation & Metric",
    "10. Conclusion",
]

selection = st.sidebar.radio("Go to:", pages)

st.sidebar.divider()
st.sidebar.caption("Built around the final ensemble pipeline and precomputed EDA summaries.")
st.sidebar.caption("Raw multi-million-row Parquet files are not loaded by the website.")


# ============================================================
# 1. OVERVIEW
# ============================================================

if selection == "1. Overview":

    st.title("AML Alert Escalation Analysis")

    st.write(
        "The goal is to estimate the probability that each alert "
        "in the hidden test set will be escalated."
    )

    st.markdown("### Our approach")

    st.write(
        "We started with exploratory data analysis of transaction "
        "history linked to alerts. We examined transaction volume, "
        "transaction amounts, incoming and outgoing activity, "
        "transaction types, recent activity and transaction history."
    )

    st.write(
        "We then converted these behavioral dimensions into numerical "
        "features, including recent-vs-historical activity, transaction "
        "anomalies, burstiness and recent transaction-size trends."
    )

    st.markdown("### What the website covers")

    st.write(
        """
        **Data:** alert-level and transaction-level structure

        **EDA:** target distribution, transaction direction and transaction types

        **Behavior:** transaction volume, amount statistics, money flow,
        transaction-type composition, recent activity and history

        **Feature engineering:** behavioral, recency, anomaly, burstiness and trend features

        **Model:** CatBoost + LightGBM + Logistic Regression ensemble with seed-bagging

        **Feature selection:** CatBoost feature-importance pruning

        **Validation:** 5-fold stratified cross-validation

        **Metric:** ROC-AUC, which evaluates ranking quality
        """
    )

    st.success(
        "The website is a presentation layer for the analysis. "
        "It does not retrain the model."
    )


# ============================================================
# 2. DATASET & STRUCTURE
# ============================================================

elif selection == "2. Dataset & Structure":

    st.title("Dataset & Structure")

    st.write(
        "The competition separates alert information from the "
        "transaction history associated with each alert."
    )

    train_alerts = total_count(target_distribution)

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Training Alerts",
        f"{train_alerts:,}" if train_alerts is not None else "Unavailable",
    )

    if test_signals is not None and "signal_id" in test_signals.columns:
        test_count = test_signals["signal_id"].nunique()
        col2.metric("Test Alerts", f"{test_count:,}")
    else:
        col2.metric("Test Alerts", "Unavailable")

    transaction_count = total_count(transaction_direction)

    col3.metric(
        "Transactions in EDA Summary",
        f"{transaction_count:,}" if transaction_count is not None else "Unavailable",
    )

    st.divider()

    st.markdown("### Alert-level dataset")

    st.dataframe(
        pd.DataFrame(
            {
                "Column": [
                    "signal_id",
                    "signal_sanasi",
                    "eskalatsiya",
                ],
                "Meaning": [
                    "Unique alert identifier",
                    "Alert date",
                    "Target: 1 = escalated, 0 = dismissed",
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### Transaction-level dataset")

    st.dataframe(
        pd.DataFrame(
            {
                "Column": [
                    "signal_id",
                    "tranzaksiya_vaqti",
                    "kirim_chiqim",
                    "tranzaksiya_turi",
                    "miqdor_indeksi",
                ],
                "Meaning": [
                    "Alert linked to the transaction",
                    "Transaction timestamp",
                    "Incoming or outgoing",
                    "Transaction category",
                    "Standardized transaction-size indicator",
                ],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### Train vs hidden test")

    st.write(
        "Training alerts contain the escalation target. Hidden test "
        "alerts do not contain the target; the model produces one "
        "escalation probability for each test alert."
    )

    st.markdown("### Leakage control")

    st.info(
        "When features were created, transactions were restricted to "
        "those occurring before the alert date. This prevents future "
        "transactions from entering the alert's feature history."
    )


# ============================================================
# 3. TARGET DISTRIBUTION
# ============================================================

elif selection == "3. Target Distribution":

    st.title("Target Distribution")

    if target_distribution is None:
        st.warning("target_distribution.csv is missing or empty.")

    elif not {"eskalatsiya", "count"}.issubset(target_distribution.columns):
        st.error(
            "target_distribution.csv must contain "
            "`eskalatsiya` and `count`."
        )

    else:
        data = target_distribution.copy()

        data["eskalatsiya"] = numeric_series(data, "eskalatsiya")
        data["count"] = numeric_series(data, "count")

        data["Target"] = data["eskalatsiya"].map(
            {
                0: "Dismissed",
                1: "Escalated",
            }
        )

        total = int(data["count"].fillna(0).sum())
        dismissed = int(
            data.loc[data["eskalatsiya"] == 0, "count"].fillna(0).sum()
        )
        escalated = int(
            data.loc[data["eskalatsiya"] == 1, "count"].fillna(0).sum()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Total", f"{total:,}")
        col2.metric("Dismissed", f"{dismissed:,}")
        col3.metric("Escalated", f"{escalated:,}")

        fig = px.bar(
            data,
            x="Target",
            y="count",
            text="count",
            title="Training Alert Outcomes",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write(
            f"Escalated alerts account for **{pct(escalated, total):.2f}%** "
            "of the training alerts."
        )

        st.markdown("### Why this matters for modeling")

        st.write(
            "The model is evaluated with ROC-AUC rather than plain "
            "accuracy. The task is to rank alerts by escalation "
            "probability, so class distribution and ranking quality "
            "must be considered together."
        )


# ============================================================
# 4. TRANSACTION BEHAVIOR
# ============================================================

elif selection == "4. Transaction Behavior":

    st.title("Transaction Behavior")

    tab1, tab2 = st.tabs(
        [
            "Direction",
            "Transaction Types",
        ]
    )

    with tab1:

        st.subheader("Incoming vs Outgoing")

        if transaction_direction is None:
            st.warning("transaction_direction.csv is missing or empty.")

        elif not {"kirim_chiqim", "count"}.issubset(
            transaction_direction.columns
        ):
            st.error(
                "transaction_direction.csv must contain "
                "`kirim_chiqim` and `count`."
            )

        else:
            data = transaction_direction.copy()
            data["count"] = numeric_series(data, "count")

            fig = px.bar(
                data,
                x="kirim_chiqim",
                y="count",
                text="count",
                title="Transaction Direction Distribution",
            )

            st.plotly_chart(fig, use_container_width=True)

            total = int(data["count"].fillna(0).sum())

            for _, row in data.iterrows():
                label = row["kirim_chiqim"]
                count = int(row["count"]) if pd.notna(row["count"]) else 0

                st.write(
                    f"**{label}**: {count:,} transactions "
                    f"({pct(count, total):.2f}%)."
                )

            st.markdown("### Modeling connection")

            st.write(
                "Direction is represented using incoming/outgoing counts "
                "and ratios. This lets the model distinguish transaction "
                "volume from the overall money-flow composition."
            )

    with tab2:

        st.subheader("Transaction Types")

        if transaction_types is None:
            st.warning("transaction_types.csv is missing or empty.")

        elif not {"tranzaksiya_turi", "count"}.issubset(
            transaction_types.columns
        ):
            st.error(
                "transaction_types.csv must contain "
                "`tranzaksiya_turi` and `count`."
            )

        else:
            data = transaction_types.copy()
            data["count"] = numeric_series(data, "count")

            fig = px.bar(
                data.sort_values("count", ascending=False),
                x="tranzaksiya_turi",
                y="count",
                text="count",
                title="Transaction Type Distribution",
            )

            st.plotly_chart(fig, use_container_width=True)

            total = int(data["count"].fillna(0).sum())

            for _, row in data.sort_values(
                "count",
                ascending=False,
            ).iterrows():

                label = row["tranzaksiya_turi"]
                count = int(row["count"]) if pd.notna(row["count"]) else 0

                st.write(
                    f"**{label}**: {count:,} transactions "
                    f"({pct(count, total):.2f}%)."
                )

            st.markdown("### Modeling connection")

            st.write(
                "Transaction-type counts and ratios preserve the composition "
                "of each alert's transaction history instead of treating all "
                "transactions as identical."
            )


# ============================================================
# 5. KEY BEHAVIORAL PATTERNS
# ============================================================

elif selection == "5. Key Behavioral Patterns":

    st.title("Key Behavioral Patterns")

    st.write(
        "The transaction history can be described through six behavioral "
        "dimensions. These are the patterns that were turned into features."
    )

    # --------------------------------------------------------
    # Pattern 1: Volume
    # --------------------------------------------------------

    st.markdown("### 1. Transaction volume")

    st.write(
        "The number of transactions linked to an alert is a basic measure "
        "of activity. We also track active days and transaction density."
    )

    st.code(
        "transaction_count\nactive_days\ntransactions_per_active_day"
    )

    # --------------------------------------------------------
    # Pattern 2: Amounts
    # --------------------------------------------------------

    st.divider()
    st.markdown("### 2. Transaction amounts")

    st.write(
        "A single average does not fully describe transaction-size behavior. "
        "The feature set therefore keeps several parts of the amount "
        "distribution."
    )

    st.code(
        "amount_sum\namount_mean\namount_std\namount_min\namount_max\n"
        "amount_median\namount quantiles"
    )

    # Optional target-specific visualization if the real file is populated.
    if alert_behavior is not None:
        amount_cols = available_columns(
            alert_behavior,
            [
                "amount_min",
                "amount_median",
                "amount_mean",
                "amount_max",
            ],
        )

        if "eskalatsiya" in alert_behavior.columns and amount_cols:

            selected = st.selectbox(
                "Explore an amount feature by target",
                amount_cols,
            )

            plot_data = alert_behavior[
                ["eskalatsiya", selected]
            ].dropna().copy()

            plot_data["Target"] = plot_data["eskalatsiya"].map(
                {
                    0: "Dismissed",
                    1: "Escalated",
                }
            )

            fig = px.box(
                plot_data,
                x="Target",
                y=selected,
                title=f"{selected} by target",
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "This comparison is shown only when a populated "
                "alert_behavior.csv is available."
            )

    # --------------------------------------------------------
    # Pattern 3: Direction
    # --------------------------------------------------------

    st.divider()
    st.markdown("### 3. Incoming vs outgoing behavior")

    st.write(
        "The feature set includes both counts and ratios for money direction."
    )

    st.code(
        "incoming_count\noutgoing_count\nincoming_ratio\noutgoing_ratio"
    )

    # --------------------------------------------------------
    # Pattern 4: Transaction types
    # --------------------------------------------------------

    st.divider()
    st.markdown("### 4. Transaction-type composition")

    st.write(
        "The history is broken into card, bank transfer, cash and "
        "international activity. Ratios allow the model to use composition."
    )

    st.code(
        "card_ratio\nbank_transfer_ratio\ncash_ratio\ninternational_ratio"
    )

    # --------------------------------------------------------
    # Pattern 5: Recent activity
    # --------------------------------------------------------

    st.divider()
    st.markdown("### 5. Recent activity")

    st.write(
        "Activity is measured across 7-, 30- and 90-day windows before the alert. "
        "The final model also compares recent activity with the account's "
        "own historical baseline."
    )

    st.code(
        "tx_last_7d\namount_last_7d\namount_mean_last_7d\n"
        "tx_last_30d\namount_last_30d\namount_mean_last_30d\n"
        "tx_last_90d\namount_last_90d\nactivity_7d_vs_30d"
    )

    # --------------------------------------------------------
    # Pattern 6: History
    # --------------------------------------------------------

    st.divider()
    st.markdown("### 6. Transaction history")

    st.write(
        "The model also receives features describing how far back "
        "transaction activity extends and how recently the last "
        "transaction occurred."
    )

    st.code(
        "history_span_days\ndays_since_last_transaction\n"
        "last_tx_zscore\nmax_to_mean_ratio\n"
        "gap_cv\nrecent_amount_trend\nactivity_vs_own_baseline"
    )

    st.info(
        "The EDA provides the foundation for the model. The final feature "
        "pipeline extends it with recency windows, account-level anomaly "
        "scores, transaction-gap variability, recent amount trends and "
        "activity relative to the account's own historical baseline."
    )


# ============================================================
# 6. EDA → FEATURE ENGINEERING
# ============================================================

elif selection == "6. EDA → Feature Engineering":

    st.title("EDA → Feature Engineering")

    st.write(
        "The main idea is simple: identify a useful behavioral dimension "
        "during EDA, represent it numerically, and then give those "
        "features to the model."
    )

    feature_table = pd.DataFrame(
        {
            "Behavior": [
                "Transaction volume",
                "Transaction amounts",
                "Money direction",
                "Transaction types",
                "Recent activity",
                "Transaction history",
                "Calendar information",
            ],
            "Features": [
                "transaction_count, active_days, "
                "transactions_per_active_day",
                "sum, mean, std, min, max, median, "
                "quantiles",
                "incoming_count, outgoing_count, "
                "incoming_ratio, outgoing_ratio",
                "card_ratio, bank_transfer_ratio, "
                "cash_ratio, international_ratio",
                "transactions_last_1d, 3d, 7d, 14d, "
                "30d, 60d, 90d, activity_7d_vs_30d",
                "history_span_days, days_since_last_transaction",
                "signal day, month, weekday and related time features",
            ],
        }
    )

    st.dataframe(
        feature_table,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### Example feature logic")

    st.write(
        "**Ratios:** convert raw counts into composition, for example "
        "outgoing transactions divided by total transactions."
    )

    st.write(
        "**Time windows:** summarize transaction count and amount behavior "
        "over 7-, 30- and 90-day windows before the alert."
    )

    st.write(
        "**Amount distribution:** use several statistics so the model "
        "can see more than one average value."
    )

    st.write(
        "**Behavior change:** compare recent activity with historical activity, "
        "measure the latest transaction's z-score, transaction-gap variability, "
        "and the recent transaction-size trend."
    )

    st.markdown("### Time leakage prevention")

    st.info(
        "For each alert, transactions were filtered using "
        "`tranzaksiya_vaqti < signal_sanasi`. This means future "
        "transaction information is excluded from the feature history."
    )

    st.markdown("### Feature matrix")

    st.write(
        "After feature engineering, `signal_id` is removed from the "
        "model inputs and `eskalatsiya` is kept only as the training target."
    )


# ============================================================
# 7. HOW THE MODEL WORKS
# ============================================================

elif selection == "7. How the Model Works":

    st.title("How the Model Works")

    st.markdown("### 1. Input")

    st.write(
        "Each alert becomes one row containing numerical behavioral "
        "features derived only from transactions before the alert."
    )

    st.markdown("### 2. Feature selection")

    st.write(
        "A quick CatBoost model ranks the engineered features by importance. "
        "The strongest cumulative importance mass is retained to reduce "
        "redundant or noisy features."
    )

    st.markdown("### 3. Three-model ensemble")

    st.write(
        "**CatBoost** captures nonlinear interactions. "
        "**LightGBM** provides a second gradient-boosting perspective. "
        "**Logistic Regression** adds a simpler linear model for diversity. "
        "Each model is trained with multiple random seeds."
    )

    st.markdown("### 4. Rank averaging")

    st.write(
        "The model outputs are converted to ranks and combined. When LightGBM "
        "is available, the final blend uses 45% CatBoost, 40% LightGBM and "
        "15% Logistic Regression. This directly supports the ROC-AUC ranking objective."
    )

    st.markdown("### 5. Final test prediction")

    st.code(
        "test alert\n"
        "    ↓\n"
        "historical transactions before alert\n"
        "    ↓\n"
        "behavioral + recency + anomaly + trend features\n"
        "    ↓\n"
        "feature-importance pruning\n"
        "    ↓\n"
        "CatBoost + LightGBM + Logistic Regression\n"
        "    ↓\n"
        "rank averaging\n"
        "    ↓\n"
        "escalation probability"
    )



# ============================================================
# 8. MODEL FEATURES
# ============================================================

elif selection == "8. Model Features":

    st.title("Model Features")

    feature_groups = pd.DataFrame(
        {
            "Feature Group": [
                "Volume",
                "Amounts",
                "Direction",
                "Transaction Types",
                "Time Windows",
                "Trend / Anomaly / Burstiness",
                "History",
                "Calendar",
            ],
            "Examples": [
                "transaction_count, active_days, "
                "transactions_per_active_day",
                "amount_sum, amount_mean, amount_std, "
                "amount_min, amount_max, amount_median, quantiles",
                "incoming_count, outgoing_count, "
                "incoming_ratio, outgoing_ratio",
                "card_ratio, bank_transfer_ratio, "
                "cash_ratio, international_ratio",
                "7d, 30d, 90d counts and amount statistics",
                "last_tx_zscore, max_to_mean_ratio, gap_cv, "
                "recent_amount_trend, activity_vs_own_baseline",
                "history_span_days, days_since_last_transaction",
                "signal month/weekday features",
            ],
        }
    )

    st.dataframe(
        feature_groups,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### Feature-importance pruning and interpretation")

    if (
        feature_importance is not None
        and {"Feature", "Importance"}.issubset(feature_importance.columns)
    ):

        importance = feature_importance.copy()
        importance["Importance"] = numeric_series(
            importance,
            "Importance",
        )

        importance = importance.sort_values(
            "Importance",
            ascending=False,
        )

        fig = px.bar(
            importance.head(20).sort_values("Importance"),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importance",
        )

        st.plotly_chart(fig, use_container_width=True)

        st.write(
            "Feature importance shows which variables the trained model "
            "used more heavily. It does not mean that a feature causes "
            "an escalation."
        )

    else:

        st.info(
            "A populated feature_importance.csv is not currently available. "
            "The website therefore describes the real feature groups instead "
            "of displaying invented importance values."
        )


# ============================================================
# 9. VALIDATION & METRIC
# ============================================================

elif selection == "9. Validation & Metric":

    st.title("Validation & Metric")

    st.markdown("### Cross-validation")

    st.write(
        "The final ensemble uses **5-fold Stratified Cross-Validation**. "
        "Each fold trains on four parts and evaluates on the held-out fifth "
        "part. Test predictions are averaged across folds and model seeds."
    )

    st.markdown("### Model configuration")

    parameters = pd.DataFrame(
        {
            "Parameter": [
                "Models",
                "Cross-validation",
                "Seeds",
                "CatBoost",
                "LightGBM",
                "Logistic Regression",
                "Ensemble weights",
                "Feature selection",
                "Final metric",
            ],
            "Value": [
                "CatBoost + LightGBM + Logistic Regression",
                "5-fold StratifiedKFold",
                "42, 202, 777",
                "2500 iterations, depth 7, learning rate 0.03, L2 = 5",
                "2000 estimators, depth 7, 63 leaves, learning rate 0.03",
                "StandardScaler + median imputation, C = 0.5, balanced classes",
                "45% CatBoost / 40% LightGBM / 15% Logistic Regression",
                "Top 85% cumulative CatBoost importance mass, minimum 15 features",
                "ROC-AUC",
            ],
        }
    )

    st.dataframe(
        parameters,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### ROC-AUC")

    st.write(
        "ROC-AUC measures ranking quality. A useful prediction system "
        "puts alerts that are actually escalated higher in the ranking "
        "than alerts that are dismissed."
    )

    st.write(
        "Because of this metric, the exact numerical calibration of the "
        "probabilities is less important than their ordering for the "
        "competition score."
    )

    if (
        model_comparison is not None
        and {"Model", "ROC_AUC"}.issubset(model_comparison.columns)
    ):

        comparison = model_comparison.copy()
        comparison["ROC_AUC"] = numeric_series(
            comparison,
            "ROC_AUC",
        )

        st.markdown("### Recorded model comparison")

        st.dataframe(
            comparison,
            hide_index=True,
            use_container_width=True,
        )

        fig = px.bar(
            comparison,
            x="Model",
            y="ROC_AUC",
            text="ROC_AUC",
            title="Recorded Validation Results",
        )

        st.plotly_chart(fig, use_container_width=True)

    else:

        st.info(
            "No populated model_comparison.csv is available, so the "
            "website does not display a fabricated comparison."
        )


# ============================================================
# 10. CONCLUSION
# ============================================================

elif selection == "10. Conclusion":

    st.title("Conclusion")

    st.write(
        "The workflow turns transaction history into an alert-level "
        "behavioral representation, selects the strongest features, and "
        "uses a seed-bagged ensemble of CatBoost, LightGBM and Logistic "
        "Regression to estimate escalation probability."
    )

    st.markdown("### What the analysis contributes")

    st.write(
        """
        **Behavioral EDA** → shows the structure of transaction activity

        **Feature engineering** → converts activity into behavioral, recency, anomaly and trend signals

        **Feature pruning** → removes lower-value or redundant columns using CatBoost importance

        **Leakage control** → uses only transactions before the alert

        **Ensemble** → combines CatBoost, LightGBM and Logistic Regression with seed-bagging

        **Rank averaging** → combines model rankings for the ROC-AUC objective

        **5-fold validation** → checks performance across multiple stratified splits

        **ROC-AUC** → evaluates how well escalation probabilities are ranked
        """
    )

    st.markdown("### Final workflow")

    st.code(
        """
Transaction history
        ↓
EDA
        ↓
Behavioral + recency + anomaly + trend features
        ↓
Feature-importance pruning
        ↓
CatBoost + LightGBM + Logistic Regression
        ↓
5-fold Stratified Cross-Validation + seed-bagging
        ↓
Rank averaging
        ↓
ROC-AUC
        ↓
Escalation probability for each test alert
        """
    )

    st.success(
        "The final submission contains one `ehtimollik` probability "
        "for every test `signal_id`."
    )
