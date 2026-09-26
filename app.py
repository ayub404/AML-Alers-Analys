import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AML Alert Escalation Analysis",
    page_icon="🔍",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

TRAIN_SIGNALS_PATH = "data/train_signals.csv"
TRAIN_TRANSACTIONS_PATH = "data/train_transactions.parquet"
TEST_SIGNALS_PATH = "data/test_signals.csv"
TEST_TRANSACTIONS_PATH = "data/test_transactions.parquet"

EDA_DIR = "eda_data"

FEATURE_FILE = os.path.join(
    EDA_DIR,
    "alert_behavior.csv"
)


# ============================================================
# LOAD RAW DATA
# ============================================================

@st.cache_data
def load_raw_data():

    train_signals = pd.read_csv(
        TRAIN_SIGNALS_PATH,
        parse_dates=["signal_sanasi"]
    )

    train_transactions = pd.read_parquet(
        TRAIN_TRANSACTIONS_PATH
    )

    test_signals = pd.read_csv(
        TEST_SIGNALS_PATH,
        parse_dates=["signal_sanasi"]
    )

    test_transactions = pd.read_parquet(
        TEST_TRANSACTIONS_PATH
    )

    # Make sure transaction time is datetime
    train_transactions["tranzaksiya_vaqti"] = pd.to_datetime(
        train_transactions["tranzaksiya_vaqti"],
        errors="coerce"
    )

    test_transactions["tranzaksiya_vaqti"] = pd.to_datetime(
        test_transactions["tranzaksiya_vaqti"],
        errors="coerce"
    )

    return (
        train_signals,
        train_transactions,
        test_signals,
        test_transactions
    )


# ============================================================
# CREATE ALERT-LEVEL BEHAVIORAL DATA
# ============================================================

@st.cache_data
def create_behavior_data(
    train_signals,
    train_transactions
):

    tx = train_transactions.merge(
        train_signals[
            [
                "signal_id",
                "signal_sanasi",
                "eskalatsiya"
            ]
        ],
        on="signal_id",
        how="inner"
    )

    # IMPORTANT:
    # Only use transactions that happened before the alert.
    tx = tx[
        tx["tranzaksiya_vaqti"] <
        tx["signal_sanasi"]
    ].copy()

    # --------------------------------------------------------
    # Time before alert
    # --------------------------------------------------------

    tx["days_before"] = (
        tx["signal_sanasi"] -
        tx["tranzaksiya_vaqti"]
    ).dt.total_seconds() / 86400

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    tx["is_in"] = (
        tx["kirim_chiqim"] == "kirim"
    ).astype("int8")

    tx["is_out"] = (
        tx["kirim_chiqim"] == "chiqim"
    ).astype("int8")

    # --------------------------------------------------------
    # Transaction types
    # --------------------------------------------------------

    tx["is_card"] = (
        tx["tranzaksiya_turi"] == "karta"
    ).astype("int8")

    tx["is_bank"] = (
        tx["tranzaksiya_turi"] == "bank_otkazmasi"
    ).astype("int8")

    tx["is_cash"] = (
        tx["tranzaksiya_turi"] == "naqd"
    ).astype("int8")

    tx["is_international"] = (
        tx["tranzaksiya_turi"] == "xalqaro"
    ).astype("int8")

    # --------------------------------------------------------
    # Main alert-level statistics
    # --------------------------------------------------------

    features = (
        tx.groupby("signal_id")
        .agg(
            transaction_count=(
                "signal_id",
                "size"
            ),

            amount_sum=(
                "miqdor_indeksi",
                "sum"
            ),

            amount_mean=(
                "miqdor_indeksi",
                "mean"
            ),

            amount_std=(
                "miqdor_indeksi",
                "std"
            ),

            amount_min=(
                "miqdor_indeksi",
                "min"
            ),

            amount_max=(
                "miqdor_indeksi",
                "max"
            ),

            amount_median=(
                "miqdor_indeksi",
                "median"
            ),

            incoming_count=(
                "is_in",
                "sum"
            ),

            outgoing_count=(
                "is_out",
                "sum"
            ),

            card_count=(
                "is_card",
                "sum"
            ),

            bank_transfer_count=(
                "is_bank",
                "sum"
            ),

            cash_count=(
                "is_cash",
                "sum"
            ),

            international_count=(
                "is_international",
                "sum"
            ),

            active_days=(
                "tranzaksiya_vaqti",
                lambda x: x.dt.date.nunique()
            ),

            first_transaction=(
                "tranzaksiya_vaqti",
                "min"
            ),

            last_transaction=(
                "tranzaksiya_vaqti",
                "max"
            )
        )
    )

    # --------------------------------------------------------
    # Quantiles
    # --------------------------------------------------------

    quantiles = (
        tx.groupby("signal_id")["miqdor_indeksi"]
        .quantile(
            [0.25, 0.50, 0.75, 0.90]
        )
        .unstack()
    )

    quantiles = quantiles.rename(
        columns={
            0.25: "amount_q25",
            0.50: "amount_q50",
            0.75: "amount_q75",
            0.90: "amount_q90"
        }
    )

    features = features.join(
        quantiles,
        how="left"
    )

    # --------------------------------------------------------
    # History length
    # --------------------------------------------------------

    features["history_span_days"] = (
        features["last_transaction"] -
        features["first_transaction"]
    ).dt.total_seconds() / 86400

    # --------------------------------------------------------
    # Time since last transaction
    # --------------------------------------------------------

    closest_transaction = (
        tx.groupby("signal_id")["days_before"]
        .min()
    )

    features["days_since_last_transaction"] = (
        closest_transaction
    )

    # --------------------------------------------------------
    # Ratios
    # --------------------------------------------------------

    transaction_count = (
        features["transaction_count"]
        .replace(0, np.nan)
    )

    features["incoming_ratio"] = (
        features["incoming_count"] /
        transaction_count
    )

    features["outgoing_ratio"] = (
        features["outgoing_count"] /
        transaction_count
    )

    features["cash_ratio"] = (
        features["cash_count"] /
        transaction_count
    )

    features["international_ratio"] = (
        features["international_count"] /
        transaction_count
    )

    features["card_ratio"] = (
        features["card_count"] /
        transaction_count
    )

    features["bank_transfer_ratio"] = (
        features["bank_transfer_count"] /
        transaction_count
    )

    features["transactions_per_active_day"] = (
        features["transaction_count"] /
        features["active_days"]
        .replace(0, np.nan)
    )

    # --------------------------------------------------------
    # Recent activity
    # --------------------------------------------------------

    for days in [1, 3, 7, 14, 30, 60, 90]:

        recent = tx[
            tx["days_before"] <= days
        ]

        if len(recent) == 0:
            continue

        recent_features = (
            recent.groupby("signal_id")
            .agg(
                **{
                    f"transactions_last_{days}d":
                        ("signal_id", "size"),

                    f"amount_last_{days}d":
                        ("miqdor_indeksi", "sum"),

                    f"outgoing_last_{days}d":
                        ("is_out", "sum"),

                    f"incoming_last_{days}d":
                        ("is_in", "sum"),

                    f"international_last_{days}d":
                        ("is_international", "sum")
                }
            )
        )

        features = features.join(
            recent_features,
            how="left"
        )

    # --------------------------------------------------------
    # Recent activity ratios
    # --------------------------------------------------------

    for days in [3, 7, 14, 30]:

        column = (
            f"transactions_last_{days}d"
        )

        if column in features.columns:

            features[
                f"recent_{days}d_ratio"
            ] = (
                features[column] /
                transaction_count
            )

    # --------------------------------------------------------
    # 7-day vs 30-day activity
    # --------------------------------------------------------

    if (
        "transactions_last_7d" in features.columns
        and
        "transactions_last_30d" in features.columns
    ):

        features[
            "activity_7d_vs_30d"
        ] = (
            features["transactions_last_7d"] /
            features[
                "transactions_last_30d"
            ].replace(0, np.nan)
        )

    # --------------------------------------------------------
    # 3-day vs 30-day activity
    # --------------------------------------------------------

    if (
        "transactions_last_3d" in features.columns
        and
        "transactions_last_30d" in features.columns
    ):

        features[
            "activity_3d_vs_30d"
        ] = (
            features["transactions_last_3d"] /
            features[
                "transactions_last_30d"
            ].replace(0, np.nan)
        )

    # --------------------------------------------------------
    # Add target
    # --------------------------------------------------------

    features = features.reset_index()

    features = features.merge(
        train_signals[
            [
                "signal_id",
                "eskalatsiya"
            ]
        ],
        on="signal_id",
        how="right"
    )

    # Replace infinity
    features = features.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return features, tx


# ============================================================
# LOAD EVERYTHING
# ============================================================

try:

    (
        train_signals,
        train_transactions,
        test_signals,
        test_transactions
    ) = load_raw_data()

except Exception as e:

    st.error(
        "Could not load the competition files."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# CREATE BEHAVIOR DATA
# ============================================================

with st.spinner(
    "Preparing behavioral analysis from the real transaction history..."
):

    df, filtered_transactions = (
        create_behavior_data(
            train_signals,
            train_transactions
        )
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

pages = [
    "1. Overview",
    "2. Dataset Overview",
    "3. Target Distribution",
    "4. Transaction Behavior",
    "5. Behavioral Patterns",
    "6. EDA → Feature Engineering",
    "7. Model",
    "8. What the Model Relied On",
    "9. Model Selection",
    "10. Conclusion"
]

selection = st.sidebar.radio(
    "Go to:",
    pages
)

st.sidebar.divider()

st.sidebar.caption(
    "Analysis is based on the competition training data."
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if selection == "1. Overview":

    st.title(
        "AML Alert Escalation Analysis"
    )

    st.write(
        "We analyzed the transaction history connected to "
        "training alerts and looked for behavioral differences "
        "between escalated and dismissed alerts."
    )

    st.markdown("### Our approach")

    st.write(
        "The analysis starts from the raw alert and transaction "
        "tables. We keep only transactions that happened before "
        "the alert date, then summarize each alert's transaction "
        "history using volume, amounts, direction, transaction "
        "types and recent activity."
    )

    st.markdown("### What we looked at")

    st.write(
        "The main areas were transaction volume, transaction "
        "amounts, incoming versus outgoing activity, payment "
        "types, history length and activity close to the alert."
    )

    st.markdown("### From EDA to the model")

    st.write(
        "The behavioral measurements from EDA were converted into "
        "numeric features. These features were then used by the "
        "CatBoost model. Model quality was evaluated using ROC-AUC."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Training Alerts",
        f"{len(train_signals):,}"
    )

    col2.metric(
        "Training Transactions",
        f"{len(train_transactions):,}"
    )

    col3.metric(
        "Test Alerts",
        f"{len(test_signals):,}"
    )


# ============================================================
# PAGE 2 — DATASET OVERVIEW
# ============================================================

elif selection == "2. Dataset Overview":

    st.title(
        "Dataset Overview"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Train Alerts",
        f"{len(train_signals):,}"
    )

    col2.metric(
        "Train Transactions",
        f"{len(train_transactions):,}"
    )

    col3.metric(
        "Test Alerts",
        f"{len(test_signals):,}"
    )

    col4.metric(
        "Test Transactions",
        f"{len(test_transactions):,}"
    )

    st.divider()

    st.subheader(
        "Training alert structure"
    )

    st.dataframe(
        train_signals.head(10),
        use_container_width=True
    )

    st.subheader(
        "Transaction structure"
    )

    st.dataframe(
        train_transactions.head(10),
        use_container_width=True
    )

    st.markdown("### Main columns")

    column_info = pd.DataFrame({
        "Dataset": [
            "train_signals",
            "train_signals",
            "train_transactions",
            "train_transactions",
            "train_transactions",
            "train_transactions",
            "train_transactions"
        ],
        "Column": [
            "signal_id",
            "eskalatsiya",
            "signal_id",
            "tranzaksiya_vaqti",
            "kirim_chiqim",
            "tranzaksiya_turi",
            "miqdor_indeksi"
        ],
        "Role": [
            "Alert identifier",
            "Target",
            "Alert identifier",
            "Transaction timestamp",
            "Incoming / outgoing direction",
            "Transaction type",
            "Standardized transaction amount indicator"
        ]
    })

    st.dataframe(
        column_info,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "For the behavioral analysis, transactions occurring "
        "after the alert date are excluded."
    )


# ============================================================
# PAGE 3 — TARGET DISTRIBUTION
# ============================================================

elif selection == "3. Target Distribution":

    st.title(
        "Target Distribution"
    )

    counts = (
        train_signals["eskalatsiya"]
        .value_counts()
        .sort_index()
    )

    dismissed = int(
        counts.get(0, 0)
    )

    escalated = int(
        counts.get(1, 0)
    )

    total = dismissed + escalated

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Dismissed",
        f"{dismissed:,}"
    )

    col2.metric(
        "Escalated",
        f"{escalated:,}"
    )

    col3.metric(
        "Escalation Rate",
        f"{escalated / total * 100:.2f}%"
        if total else "0%"
    )

    target_df = pd.DataFrame({
        "Target": [
            "Dismissed (0)",
            "Escalated (1)"
        ],
        "Count": [
            dismissed,
            escalated
        ]
    })

    fig = px.bar(
        target_df,
        x="Target",
        y="Count",
        title="Training Alert Target Distribution",
        text="Count"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write(
        "ROC-AUC is used for evaluation because the competition "
        "focuses on ranking alerts by escalation probability."
    )


# ============================================================
# PAGE 4 — TRANSACTION BEHAVIOR
# ============================================================

elif selection == "4. Transaction Behavior":

    st.title(
        "Transaction Behavior"
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Incoming vs Outgoing",
            "Transaction Types",
            "Amounts",
            "Activity"
        ]
    )

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    with tab1:

        st.subheader(
            "Incoming vs Outgoing Transactions"
        )

        direction = pd.DataFrame({
            "Direction": [
                "Incoming",
                "Outgoing"
            ],
            "Count": [
                int(
                    filtered_transactions["is_in"].sum()
                ),
                int(
                    filtered_transactions["is_out"].sum()
                )
            ]
        })

        fig = px.bar(
            direction,
            x="Direction",
            y="Count",
            title="Transaction Direction"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Outgoing Ratio by Target"
        )

        plot_df = df[
            [
                "signal_id",
                "eskalatsiya",
                "outgoing_ratio"
            ]
        ].dropna()

        plot_df["Target"] = plot_df[
            "eskalatsiya"
        ].map({
            0: "Dismissed",
            1: "Escalated"
        })

        fig = px.histogram(
            plot_df,
            x="outgoing_ratio",
            color="Target",
            barmode="overlay",
            nbins=30,
            opacity=0.7,
            title="Distribution of Outgoing Transaction Ratio"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Types
    # --------------------------------------------------------

    with tab2:

        st.subheader(
            "Transaction Types"
        )

        type_counts = (
            filtered_transactions[
                "tranzaksiya_turi"
            ]
            .value_counts()
            .reset_index()
        )

        type_counts.columns = [
            "Transaction Type",
            "Count"
        ]

        fig = px.bar(
            type_counts,
            x="Transaction Type",
            y="Count",
            title="Transaction Types in Training History",
            text="Count"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Transaction Type Ratios by Alert"
        )

        type_ratio_data = df[
            [
                "card_ratio",
                "bank_transfer_ratio",
                "cash_ratio",
                "international_ratio"
            ]
        ].mean().reset_index()

        type_ratio_data.columns = [
            "Feature",
            "Average Ratio"
        ]

        fig = px.bar(
            type_ratio_data,
            x="Feature",
            y="Average Ratio",
            title="Average Transaction Type Ratio per Alert"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Amounts
    # --------------------------------------------------------

    with tab3:

        st.subheader(
            "Transaction Amount Distribution"
        )

        amount_plot = df[
            [
                "eskalatsiya",
                "amount_mean"
            ]
        ].dropna()

        amount_plot["Target"] = amount_plot[
            "eskalatsiya"
        ].map({
            0: "Dismissed",
            1: "Escalated"
        })

        fig = px.box(
            amount_plot,
            x="Target",
            y="amount_mean",
            color="Target",
            title="Mean Transaction Amount by Target"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Minimum Transaction Amount"
        )

        amount_min_plot = df[
            [
                "eskalatsiya",
                "amount_min"
            ]
        ].dropna()

        amount_min_plot["Target"] = (
            amount_min_plot["eskalatsiya"]
            .map({
                0: "Dismissed",
                1: "Escalated"
            })
        )

        fig = px.box(
            amount_min_plot,
            x="Target",
            y="amount_min",
            color="Target",
            title="Minimum Transaction Amount by Target"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Activity
    # --------------------------------------------------------

    with tab4:

        st.subheader(
            "Transaction Count per Alert"
        )

        count_plot = df[
            [
                "eskalatsiya",
                "transaction_count"
            ]
        ].dropna()

        count_plot["Target"] = (
            count_plot["eskalatsiya"]
            .map({
                0: "Dismissed",
                1: "Escalated"
            })
        )

        fig = px.box(
            count_plot,
            x="Target",
            y="transaction_count",
            color="Target",
            title="Transaction Count by Target"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# PAGE 5 — BEHAVIORAL PATTERNS
# ============================================================

elif selection == "5. Behavioral Patterns":

    st.title(
        "Behavioral Patterns We Found"
    )

    st.write(
        "The charts below compare measurable transaction behavior "
        "between dismissed and escalated training alerts. "
        "These are descriptive patterns, not causal conclusions."
    )

    # --------------------------------------------------------
    # Pattern 1
    # --------------------------------------------------------

    st.markdown(
        "### 1. Transaction Volume"
    )

    col1, col2 = st.columns(2)

    with col1:

        plot_df = df[
            [
                "eskalatsiya",
                "transaction_count"
            ]
        ].dropna()

        plot_df["Target"] = plot_df[
            "eskalatsiya"
        ].map({
            0: "Dismissed",
            1: "Escalated"
        })

        fig = px.box(
            plot_df,
            x="Target",
            y="transaction_count",
            color="Target",
            title="Transaction Count per Alert"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        summary = (
            df.groupby("eskalatsiya")
            ["transaction_count"]
            .median()
            .reset_index()
        )

        summary["Target"] = summary[
            "eskalatsiya"
        ].map({
            0: "Dismissed",
            1: "Escalated"
        })

        st.write(
            "The median transaction count is shown below."
        )

        st.dataframe(
            summary[
                ["Target", "transaction_count"]
            ].rename(
                columns={
                    "transaction_count":
                        "Median Transactions"
                }
            ),
            hide_index=True,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Pattern 2
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "### 2. Amount Behavior"
    )

    col1, col2 = st.columns(2)

    with col1:

        plot_df = df[
            [
                "eskalatsiya",
                "amount_min"
            ]
        ].dropna()

        plot_df["Target"] = (
            plot_df["eskalatsiya"]
            .map({
                0: "Dismissed",
                1: "Escalated"
            })
        )

        fig = px.box(
            plot_df,
            x="Target",
            y="amount_min",
            color="Target",
            title="Minimum Transaction Amount"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.write(
            "Minimum, maximum, mean and median transaction "
            "amounts are calculated for each alert. This gives "
            "the model information about both typical and extreme "
            "transaction sizes."
        )

        amount_summary = (
            df.groupby("eskalatsiya")[
                [
                    "amount_min",
                    "amount_median",
                    "amount_mean",
                    "amount_max"
                ]
            ]
            .median()
            .reset_index()
        )

        amount_summary["Target"] = (
            amount_summary["eskalatsiya"]
            .map({
                0: "Dismissed",
                1: "Escalated"
            })
        )

        st.dataframe(
            amount_summary[
                [
                    "Target",
                    "amount_min",
                    "amount_median",
                    "amount_mean",
                    "amount_max"
                ]
            ].rename(
                columns={
                    "amount_min": "Median Min",
                    "amount_median": "Median Amount",
                    "amount_mean": "Mean Amount",
                    "amount_max": "Median Max"
                }
            ),
            hide_index=True,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Pattern 3
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "### 3. Incoming vs Outgoing Behavior"
    )

    col1, col2 = st.columns(2)

    with col1:

        plot_df = df[
            [
                "eskalatsiya",
                "outgoing_ratio"
            ]
        ].dropna()

        plot_df["Target"] = (
            plot_df["eskalatsiya"]
            .map({
                0: "Dismissed",
                1: "Escalated"
            })
        )

        fig = px.box(
            plot_df,
            x="Target",
            y="outgoing_ratio",
            color="Target",
            title="Outgoing Transaction Ratio"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.write(
            "Incoming and outgoing transaction counts are "
            "converted into ratios so alerts with different "
            "amounts of transaction history can still be compared."
        )

        direction_summary = (
            df.groupby("eskalatsiya")[
                [
                    "incoming_ratio",
                    "outgoing_ratio"
                ]
            ]
            .median()
            .reset_index()
        )

        direction_summary["Target"] = (
            direction_summary["eskalatsiya"]
            .map({
                0: "Dismissed",
                1: "Escalated"
            })
        )

        st.dataframe(
            direction_summary[
                [
                    "Target",
                    "incoming_ratio",
                    "outgoing_ratio"
                ]
            ].rename(
                columns={
                    "incoming_ratio":
                        "Median Incoming Ratio",
                    "outgoing_ratio":
                        "Median Outgoing Ratio"
                }
            ),
            hide_index=True,
            use_container_width=True
        )

    # --------------------------------------------------------
    # Pattern 4
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "### 4. Recent Activity"
    )

    if "activity_7d_vs_30d" in df.columns:

        plot_df = df[
            [
                "eskalatsiya",
                "activity_7d_vs_30d"
            ]
        ].dropna()

        plot_df["Target"] = (
            plot_df["eskalatsiya"]
            .map({
                0: "Dismissed",
                1: "Escalated"
            })
        )

        fig = px.box(
            plot_df,
            x="Target",
            y="activity_7d_vs_30d",
            color="Target",
            title="7-Day Activity Relative to 30-Day Activity"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.write(
            "This ratio compares recent activity with the "
            "longer 30-day window. A higher value means a larger "
            "share of the alert's recent transactions happened "
            "during the last seven days."
        )

    # --------------------------------------------------------
    # Pattern 5
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        "### 5. Transaction History"
    )

    history_plot = df[
        [
            "eskalatsiya",
            "history_span_days"
        ]
    ].dropna()

    history_plot["Target"] = (
        history_plot["eskalatsiya"]
        .map({
            0: "Dismissed",
            1: "Escalated"
        })
    )

    fig = px.box(
        history_plot,
        x="Target",
        y="history_span_days",
        color="Target",
        title="Transaction History Span"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.write(
        "History span measures the time between the first and "
        "last transaction available before the alert."
    )


# ============================================================
# PAGE 6 — EDA → FEATURE ENGINEERING
# ============================================================

elif selection == "6. EDA → Feature Engineering":

    st.title(
        "From EDA to Feature Engineering"
    )

    st.write(
        "The purpose of EDA was not only to create charts. "
        "We converted measurable transaction behavior into "
        "features that could be used by the model."
    )

    feature_table = pd.DataFrame({

        "Behavior": [
            "Transaction volume",
            "Transaction amounts",
            "Money direction",
            "Transaction types",
            "Recent activity",
            "Transaction history"
        ],

        "Features": [
            "`transaction_count`, `transactions_per_active_day`",

            "`amount_sum`, `amount_mean`, `amount_min`, "
            "`amount_max`, `amount_median`, quantiles",

            "`incoming_count`, `outgoing_count`, "
            "`incoming_ratio`, `outgoing_ratio`",

            "`card_ratio`, `bank_transfer_ratio`, "
            "`cash_ratio`, `international_ratio`",

            "`transactions_last_1d`, `3d`, `7d`, `14d`, "
            "`30d`, `60d`, `90d`, `activity_7d_vs_30d`",

            "`active_days`, `history_span_days`, "
            "`days_since_last_transaction`"
        ]
    })

    st.dataframe(
        feature_table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown(
        "### Important modeling rule"
    )

    st.info(
        "Only transaction history available before the alert "
        "date is used. This prevents transactions after the "
        "alert from entering the features."
    )


# ============================================================
# PAGE 7 — MODEL
# ============================================================

elif selection == "7. Model":

    st.title(
        "Model"
    )

    st.markdown(
        "### Model used"
    )

    st.write(
        "**CatBoostClassifier**"
    )

    st.write(
        "The final model uses the engineered transaction-history "
        "features described in the previous sections."
    )

    st.markdown(
        "### Validation"
    )

    st.write(
        "**5-fold Stratified Cross-Validation**"
    )

    st.write(
        "The training alerts were divided into five stratified "
        "folds. Each fold was used as validation once while the "
        "other folds were used for training."
    )

    st.markdown(
        "### Evaluation metric"
    )

    st.write(
        "**ROC-AUC**"
    )

    st.write(
        "ROC-AUC measures how well the model ranks escalated "
        "alerts above dismissed alerts. This matches the "
        "competition evaluation metric."
    )

    st.markdown(
        "### Final model configuration"
    )

    model_params = pd.DataFrame({
        "Parameter": [
            "iterations",
            "learning_rate",
            "depth",
            "l2_leaf_reg",
            "loss_function",
            "eval_metric",
            "early_stopping_rounds"
        ],

        "Value": [
            2500,
            0.03,
            7,
            5,
            "Logloss",
            "AUC",
            150
        ]
    })

    st.dataframe(
        model_params,
        hide_index=True,
        use_container_width=True
    )


# ============================================================
# PAGE 8 — MODEL FEATURE IMPORTANCE
# ============================================================

elif selection == "8. What the Model Relied On":

    st.title(
        "What the Model Relied On"
    )

    st.warning(
        "Feature importance tells us which variables the model "
        "used heavily. It does not prove that a feature causes "
        "an alert to be escalated."
    )

    importance_path = os.path.join(
        EDA_DIR,
        "feature_importance.csv"
    )

    if os.path.exists(importance_path):

        importance = pd.read_csv(
            importance_path
        )

        required_columns = {
            "Feature",
            "Importance"
        }

        if required_columns.issubset(
            importance.columns
        ):

            importance = (
                importance
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .head(20)
            )

            fig = px.bar(
                importance.sort_values(
                    "Importance"
                ),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance from the Final Model"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.dataframe(
                importance,
                hide_index=True,
                use_container_width=True
            )

        else:

            st.error(
                "feature_importance.csv does not contain "
                "Feature and Importance columns."
            )

    else:

        st.info(
            "The website is using real EDA data, but a saved "
            "feature_importance.csv has not been added yet. "
            "This section intentionally does not invent "
            "importance values."
        )

        st.write(
            "Create feature_importance.csv from the final "
            "CatBoost model if you want the exact model "
            "importance chart here."
        )


# ============================================================
# PAGE 9 — MODEL SELECTION
# ============================================================

elif selection == "9. Model Selection":

    st.title(
        "Model Selection"
    )

    st.write(
        "We tested an expanded V2 feature set containing "
        "additional behavioral features. The final choice was "
        "based on validation ROC-AUC rather than on how many "
        "features were available."
    )

    results_path = os.path.join(
        EDA_DIR,
        "model_comparison.csv"
    )

    if os.path.exists(results_path):

        results = pd.read_csv(
            results_path
        )

        st.dataframe(
            results,
            hide_index=True,
            use_container_width=True
        )

        if {
            "Model",
            "ROC_AUC"
        }.issubset(results.columns):

            fig = px.bar(
                results,
                x="Model",
                y="ROC_AUC",
                title="Validation ROC-AUC Comparison",
                text="ROC_AUC"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    else:

        st.info(
            "No model_comparison.csv was provided to the "
            "website, so validation scores are not hard-coded "
            "here."
        )

        st.write(
            "The final model should be reported using the actual "
            "cross-validation results from the reproducible "
            "training notebook."
        )


# ============================================================
# PAGE 10 — CONCLUSION
# ============================================================

elif selection == "10. Conclusion":

    st.title(
        "Conclusion"
    )

    st.write(
        "The analysis shows how transaction history can be "
        "converted into alert-level behavioral features."
    )

    st.markdown(
        """
        **The analysis workflow was:**

        Raw transaction history  
        ↓  
        Keep transactions before the alert  
        ↓  
        Analyze transaction behavior  
        ↓  
        Create volume, amount, direction, type and time-window features  
        ↓  
        Train CatBoost  
        ↓  
        Evaluate using 5-fold ROC-AUC  
        ↓  
        Generate escalation probabilities for the hidden test set
        """
    )

    st.divider()

    st.markdown(
        "### Main behavioral areas analyzed"
    )

    st.write(
        "- Transaction volume"
    )

    st.write(
        "- Transaction amount distribution"
    )

    st.write(
        "- Incoming vs outgoing activity"
    )

    st.write(
        "- Transaction type composition"
    )

    st.write(
        "- Recent transaction activity"
    )

    st.write(
        "- Transaction history length"
    )

    st.write(
        "- Time between the latest transaction and the alert"
    )

    st.success(
        "All EDA charts on this website are calculated from "
        "the real competition training data."
    )