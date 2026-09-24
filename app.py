
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix,
    mean_absolute_error, mean_squared_error, r2_score
)

st.set_page_config(
    page_title="Sleep Health ML App",
    page_icon="😴",
    layout="wide"
)

st.title("😴 Sleep Health & Daily Performance")
st.caption("Machine Learning application based on the uploaded Sleep Health project.")

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_csv(file_path):
    return pd.read_csv(file_path)


def validate_dataset(df):
    required = {
        "sleep_disorder_risk",
        "sleep_quality_score",
        "felt_rested"
    }

    missing = required - set(df.columns)

    if missing:
        st.error(
            f"Missing required columns: {', '.join(sorted(missing))}"
        )
        st.stop()


# Your dataset path
file_path = r"resources/sleep_health_dataset.csv"

df = load_csv(file_path)

validate_dataset(df)


# Remove helper column if it exists
if "sleep_disorder_num" in df.columns:
    df = df.drop(columns=["sleep_disorder_num"])


st.write("Dataset loaded successfully!")
# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Application")

task = st.sidebar.radio(
    "Choose prediction task",
    [
        "Sleep Disorder Risk Classification",
        "Felt Rested Classification",
        "Sleep Quality Score Regression"
    ]
)

test_size = st.sidebar.slider(
    "Test size",
    min_value=0.10,
    max_value=0.40,
    value=0.20,
    step=0.05
)

random_state = 42

# -----------------------------
# Dataset overview
# -----------------------------
with st.expander("📊 Dataset Overview", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isna().sum().sum()))

    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Column Information")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.astype(str).values,
        "Missing": df.isna().sum().values,
        "Unique": df.nunique().values
    })
    st.dataframe(info_df, use_container_width=True)

# -----------------------------
# Helper functions
# -----------------------------
def build_preprocessor(X):
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ])

    return preprocessor

def get_feature_names(fitted_pipeline):
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out()

# -----------------------------
# CLASSIFICATION
# -----------------------------
if task == "Sleep Disorder Risk Classification":

    st.header("🩺 Sleep Disorder Risk Classification")
    st.write(
        "Predicts the sleep disorder risk category using Decision Tree "
        "and Random Forest classifiers."
    )

    X = df.drop(columns=["sleep_disorder_risk"], errors="ignore")
    y_text = df["sleep_disorder_risk"].astype(str)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    preprocessor = build_preprocessor(X)

    # Notebook's tuned Decision Tree settings
    dt_model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", DecisionTreeClassifier(
            criterion="entropy",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=10,
            random_state=random_state
        ))
    ])

    rf_model = Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1
        ))
    ])

    with st.spinner("Training Decision Tree and Random Forest..."):
        dt_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)

    pred_dt = dt_model.predict(X_test)
    pred_rf = rf_model.predict(X_test)

    dt_acc = accuracy_score(y_test, pred_dt)
    rf_acc = accuracy_score(y_test, pred_rf)

    results = pd.DataFrame({
        "Model": ["Decision Tree", "Random Forest"],
        "Accuracy": [dt_acc, rf_acc],
        "Precision": [
            precision_score(y_test, pred_dt, average="weighted", zero_division=0),
            precision_score(y_test, pred_rf, average="weighted", zero_division=0)
        ],
        "Recall": [
            recall_score(y_test, pred_dt, average="weighted", zero_division=0),
            recall_score(y_test, pred_rf, average="weighted", zero_division=0)
        ],
        "F1 Score": [
            f1_score(y_test, pred_dt, average="weighted", zero_division=0),
            f1_score(y_test, pred_rf, average="weighted", zero_division=0)
        ]
    })

    best_idx = results["Accuracy"].idxmax()
    best_name = results.loc[best_idx, "Model"]

    st.subheader("Model Comparison")

    cols = st.columns(2)
    cols[0].metric("Decision Tree Accuracy", f"{dt_acc * 100:.2f}%")
    cols[1].metric("Random Forest Accuracy", f"{rf_acc * 100:.2f}%")

    st.dataframe(
        results.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}"
        }),
        use_container_width=True
    )

    st.success(f"🏆 Best model: **{best_name}**")

    # Confusion matrix
    st.subheader("Confusion Matrix")

    selected_model = st.selectbox(
        "Select model",
        ["Decision Tree", "Random Forest"]
    )

    selected_pred = pred_dt if selected_model == "Decision Tree" else pred_rf

    cm = confusion_matrix(y_test, selected_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{selected_model} Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)

    # Feature importance
    st.subheader("Top 15 Important Features")

    fitted = dt_model if selected_model == "Decision Tree" else rf_model
    model_step = fitted.named_steps["model"]
    feature_names = get_feature_names(fitted)

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model_step.feature_importances_
    }).sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(
        data=importance,
        x="Importance",
        y="Feature",
        ax=ax
    )
    ax.set_title(f"Top 15 {selected_model} Features")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")
    report = classification_report(
        y_test,
        selected_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )
    st.dataframe(
        pd.DataFrame(report).transpose(),
        use_container_width=True
    )

    # Prediction form
    st.subheader("🔮 Predict a New Person")

    feature_columns = list(X.columns)

    # Create a simple form dynamically.
    # For categorical fields, use dataset values; for numeric fields, use median.
    with st.form("classification_form"):
        values = {}

        form_cols = st.columns(2)

        for i, col in enumerate(feature_columns):
            if col == "person_id":
                # ID is not useful for prediction, but the notebook keeps it.
                default = int(df[col].median())
                values[col] = form_cols[i % 2].number_input(
                    col, value=default, step=1
                )

            elif pd.api.types.is_numeric_dtype(df[col]):
                series = pd.to_numeric(df[col], errors="coerce")
                default = float(series.median())
                min_v = float(series.min())
                max_v = float(series.max())

                values[col] = form_cols[i % 2].number_input(
                    col,
                    min_value=min_v,
                    max_value=max_v,
                    value=default
                )

            else:
                options = sorted(df[col].dropna().astype(str).unique().tolist())
                values[col] = form_cols[i % 2].selectbox(
                    col,
                    options if options else [""]
                )

        submitted = st.form_submit_button("Predict Risk")

    if submitted:
        new_person = pd.DataFrame([values])
        model_to_use = dt_model if selected_model == "Decision Tree" else rf_model

        prediction_encoded = model_to_use.predict(new_person)[0]
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]

        probabilities = model_to_use.predict_proba(new_person)[0]
        confidence = float(np.max(probabilities))

        st.success(f"Predicted Sleep Disorder Risk: **{prediction}**")
        st.metric("Model Confidence", f"{confidence * 100:.2f}%")

# -----------------------------
# FELT RESTED CLASSIFICATION
# -----------------------------
elif task == "Felt Rested Classification":

    st.header("😊 Felt Rested Classification")
    st.write(
        "Predicts whether a person feels rested after sleep using "
        "the Random Forest classifier, following the notebook."
    )

    TARGET = "felt_rested"

    if TARGET not in df.columns:
        st.error("The uploaded dataset does not contain the 'felt_rested' column.")
        st.stop()

    X = df.drop(columns=[TARGET]).copy()
    y = pd.to_numeric(df[TARGET], errors="coerce")

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Same Random Forest setup used in the notebook.
    rf_felt_model = Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", RandomForestClassifier(
            n_estimators=100, random_state=random_state, n_jobs=-1
        ))
    ])

    with st.spinner("Training Random Forest for Felt Rested..."):
        rf_felt_model.fit(X_train, y_train)

    pred = rf_felt_model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, pred, average="weighted", zero_division=0)

    st.subheader("Model Performance")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Accuracy", f"{accuracy * 100:.2f}%")
    metric_cols[1].metric("Precision", f"{precision * 100:.2f}%")
    metric_cols[2].metric("Recall", f"{recall * 100:.2f}%")
    metric_cols[3].metric("F1 Score", f"{f1 * 100:.2f}%")

    results = pd.DataFrame({
        "Model": ["Random Forest"],
        "Accuracy": [accuracy],
        "Precision": [precision],
        "Recall": [recall],
        "F1 Score": [f1]
    })
    st.dataframe(results.style.format({
        "Accuracy": "{:.2%}", "Precision": "{:.2%}",
        "Recall": "{:.2%}", "F1 Score": "{:.2%}"
    }), use_container_width=True)

    st.success(f"🏆 Random Forest Accuracy: **{accuracy * 100:.2f}%**")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Not Rested (0)", "Felt Rested (1)"],
        yticklabels=["Not Rested (0)", "Felt Rested (1)"], ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Random Forest - Felt Rested Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Top 15 Important Features")
    model_step = rf_felt_model.named_steps["model"]
    feature_names = get_feature_names(rf_felt_model)
    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model_step.feature_importances_
    }).sort_values("Importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(data=importance, x="Importance", y="Feature", ax=ax)
    ax.set_title("Top 15 Random Forest Features - Felt Rested")
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Classification Report")
    report = classification_report(
        y_test, pred, target_names=["Not Rested", "Felt Rested"],
        output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    st.subheader("🔮 Predict Whether a New Person Feels Rested")
    feature_columns = list(X.columns)

    with st.form("felt_rested_form"):
        values = {}
        form_cols = st.columns(2)
        for i, col in enumerate(feature_columns):
            if col == "person_id":
                default = int(pd.to_numeric(df[col], errors="coerce").median())
                values[col] = form_cols[i % 2].number_input(col, value=default, step=1)
            elif pd.api.types.is_numeric_dtype(df[col]):
                series = pd.to_numeric(df[col], errors="coerce")
                default = float(series.median())
                min_v = float(series.min())
                max_v = float(series.max())
                values[col] = form_cols[i % 2].number_input(
                    col, min_value=min_v, max_value=max_v, value=default
                )
            else:
                options = sorted(df[col].dropna().astype(str).unique().tolist())
                values[col] = form_cols[i % 2].selectbox(col, options if options else [""])

        submitted = st.form_submit_button("Predict Felt Rested")

    if submitted:
        new_person = pd.DataFrame([values])
        prediction = int(rf_felt_model.predict(new_person)[0])
        probabilities = rf_felt_model.predict_proba(new_person)[0]
        confidence = float(np.max(probabilities))

        if prediction == 1:
            st.success("😊 Prediction: **Felt Rested**")
        else:
            st.warning("😴 Prediction: **Not Rested**")
        st.metric("Model Confidence", f"{confidence * 100:.2f}%")

# -----------------------------
# REGRESSION
# -----------------------------
else:

    st.header("📈 Sleep Quality Score Regression")
    st.write(
        "Predicts the sleep quality score using Linear Regression, "
        "Decision Tree Regression, and Random Forest Regression."
    )

    TARGET = "sleep_quality_score"

    # Follow the notebook's leakage exclusions.
    leakage_columns = [
        TARGET,
        "felt_rested",
        "sleep_disorder_risk"
    ]

    X = df.drop(
        columns=[c for c in leakage_columns if c in df.columns]
    ).copy()
    y = pd.to_numeric(df[TARGET], errors="coerce")

    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    def regression_pipeline(model):
        return Pipeline([
            ("preprocessor", build_preprocessor(X)),
            ("model", model)
        ])

    linear_model = regression_pipeline(LinearRegression())

    tree_model = regression_pipeline(
        DecisionTreeRegressor(
            max_depth=10,
            random_state=random_state
        )
    )

    forest_model = regression_pipeline(
        RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=random_state,
            n_jobs=-1
        )
    )

    models = {
        "Linear Regression": linear_model,
        "Decision Tree": tree_model,
        "Random Forest": forest_model
    }

    predictions = {}
    regression_rows = []

    with st.spinner("Training regression models..."):
        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            predictions[name] = pred

            mae = mean_absolute_error(y_test, pred)
            rmse = np.sqrt(mean_squared_error(y_test, pred))
            r2 = r2_score(y_test, pred)

            # This is the same accuracy-style calculation used in the notebook.
            accuracy = 100 - ((mae / y_test.mean()) * 100)

            regression_rows.append({
                "Model": name,
                "MAE": mae,
                "RMSE": rmse,
                "R2 Score": r2,
                "Accuracy": accuracy
            })

    results = pd.DataFrame(regression_rows)
    best_idx = results["Accuracy"].idxmax()
    best_name = results.loc[best_idx, "Model"]

    st.subheader("Model Comparison")

    st.dataframe(
        results.style.format({
            "MAE": "{:.4f}",
            "RMSE": "{:.4f}",
            "R2 Score": "{:.4f}",
            "Accuracy": "{:.2f}%"
        }),
        use_container_width=True
    )

    metric_cols = st.columns(3)
    metric_cols[0].metric(
        "Best Model",
        best_name
    )
    metric_cols[1].metric(
        "Best R²",
        f"{results.loc[best_idx, 'R2 Score']:.4f}"
    )
    metric_cols[2].metric(
        "Best Accuracy",
        f"{results.loc[best_idx, 'Accuracy']:.2f}%"
    )

    st.success(f"🏆 Best model: **{best_name}**")

    # Accuracy chart
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(results["Model"], results["Accuracy"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Model")
    ax.set_title("Accuracy Comparison")
    ax.set_ylim(0, 100)

    for i, value in enumerate(results["Accuracy"]):
        ax.text(i, value, f"{value:.2f}%", ha="center", va="bottom")

    st.pyplot(fig)
    plt.close(fig)

    # Actual vs predicted
    st.subheader("Actual vs Predicted")

    selected_model = st.selectbox(
        "Select regression model",
        list(models.keys()),
        index=list(models.keys()).index(best_name)
    )

    selected_prediction = predictions[selected_model]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_test, selected_prediction, alpha=0.4)

    min_value = min(y_test.min(), selected_prediction.min())
    max_value = max(y_test.max(), selected_prediction.max())

    ax.plot(
        [min_value, max_value],
        [min_value, max_value],
        linestyle="--"
    )

    ax.set_xlabel("Actual Sleep Quality Score")
    ax.set_ylabel("Predicted Sleep Quality Score")
    ax.set_title(f"Actual vs Predicted - {selected_model}")

    st.pyplot(fig)
    plt.close(fig)

    # Regression prediction form
    st.subheader("🔮 Predict Sleep Quality for a New Person")

    feature_columns = list(X.columns)

    with st.form("regression_form"):
        values = {}
        form_cols = st.columns(2)

        for i, col in enumerate(feature_columns):
            if col == "person_id":
                default = int(df[col].median())
                values[col] = form_cols[i % 2].number_input(
                    col, value=default, step=1
                )

            elif pd.api.types.is_numeric_dtype(df[col]):
                series = pd.to_numeric(df[col], errors="coerce")
                default = float(series.median())
                min_v = float(series.min())
                max_v = float(series.max())

                values[col] = form_cols[i % 2].number_input(
                    col,
                    min_value=min_v,
                    max_value=max_v,
                    value=default
                )

            else:
                options = sorted(df[col].dropna().astype(str).unique().tolist())
                values[col] = form_cols[i % 2].selectbox(
                    col,
                    options if options else [""]
                )

        submitted = st.form_submit_button("Predict Sleep Quality")

    if submitted:
        new_person = pd.DataFrame([values])
        selected_pipeline = models[selected_model]

        prediction = float(selected_pipeline.predict(new_person)[0])

        st.success(
            f"Predicted Sleep Quality Score: **{prediction:.2f} / 10**"
        )

        if prediction >= 7:
            st.info("This prediction indicates relatively good sleep quality.")
        elif prediction >= 4:
            st.warning("This prediction indicates moderate sleep quality.")
        else:
            st.error("This prediction indicates relatively low sleep quality.")

# -----------------------------
# Footer
# -----------------------------
st.divider()
st.caption(
    "Educational machine-learning project. Predictions are model outputs "
    "and should not be treated as medical diagnosis."
)
