from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


REQUIRED_COLUMNS = [
    "Component",
    "Sub-component",
    "Weight",
    "Primary Material",
    "% Primary Material",
    "Secondary Material",
    "% Secondary Material",
    "Lifespan",
    "Connection Method",
]

NUMERIC_COLUMNS = [
    "Weight",
    "% Primary Material",
    "% Secondary Material",
    "Lifespan",
]

CATEGORICAL_COLUMNS = [
    "Component",
    "Sub-component",
    "Primary Material",
    "Secondary Material",
    "Connection Method",
]

MODEL_PATH = Path("models/decision_tree_model.joblib")
TRAINING_DATA_PATH = Path("Full_Traning_Set.csv")
TARGET_COLUMN = "Target_Class"
PREDICTION_COLUMN = "Classification"
CONFIDENCE_COLUMN = "Confidence"


@dataclass
class ValidationResult:
    is_valid: bool
    cleaned_data: pd.DataFrame
    errors: list[str]
    warnings: list[str]


def configure_page() -> None:
    st.set_page_config(
        page_title="Component Classification Dashboard",
        page_icon="",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; }
        div[data-testid="stMetric"] {
            border: 1px solid #d7dde5;
            border-radius: 8px;
            padding: 12px 14px;
            background: #fbfcfd;
        }
        .status-row {
            border: 1px solid #d7dde5;
            border-radius: 8px;
            padding: 14px 16px;
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(column).strip() for column in cleaned.columns]
    return cleaned


def read_excel(uploaded_file: Any) -> pd.DataFrame:
    return normalize_columns(pd.read_excel(uploaded_file, engine="openpyxl"))


def validate_data(df: pd.DataFrame) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    cleaned = normalize_columns(df)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in cleaned.columns]
    if missing_columns:
        errors.append("Missing required columns: " + ", ".join(missing_columns))
        return ValidationResult(False, cleaned, errors, warnings)

    cleaned = cleaned[REQUIRED_COLUMNS].copy()
    empty_rows = cleaned.index[cleaned.isna().all(axis=1)].tolist()
    if empty_rows:
        warnings.append(f"Removed {len(empty_rows)} completely empty row(s).")
        cleaned = cleaned.drop(index=empty_rows)

    if cleaned.empty:
        errors.append("The uploaded file does not contain any usable data rows.")
        return ValidationResult(False, cleaned, errors, warnings)

    for column in REQUIRED_COLUMNS:
        missing_mask = cleaned[column].isna() | (cleaned[column].astype(str).str.strip() == "")
        if missing_mask.any():
            rows = [str(index + 2) for index in cleaned.index[missing_mask].tolist()]
            errors.append(f"Column '{column}' has missing value(s) in Excel row(s): {', '.join(rows)}")

    for column in NUMERIC_COLUMNS:
        values = pd.to_numeric(cleaned[column], errors="coerce")
        invalid_mask = values.isna()
        if invalid_mask.any():
            rows = [str(index + 2) for index in cleaned.index[invalid_mask].tolist()]
            errors.append(f"Column '{column}' must be numeric in Excel row(s): {', '.join(rows)}")
        cleaned[column] = values

    percentage_total = cleaned["% Primary Material"] + cleaned["% Secondary Material"]
    bad_percentage_mask = percentage_total.round(2) > 100
    if bad_percentage_mask.any():
        rows = [str(index + 2) for index in cleaned.index[bad_percentage_mask].tolist()]
        errors.append(
            "Primary and secondary material percentages cannot exceed 100 in Excel row(s): "
            + ", ".join(rows)
        )

    negative_numeric = cleaned[NUMERIC_COLUMNS].lt(0).any(axis=1)
    if negative_numeric.any():
        rows = [str(index + 2) for index in cleaned.index[negative_numeric].tolist()]
        errors.append("Numeric values cannot be negative in Excel row(s): " + ", ".join(rows))

    return ValidationResult(len(errors) == 0, cleaned, errors, warnings)


def prepare_data_for_model(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df[REQUIRED_COLUMNS].copy()

    for column in NUMERIC_COLUMNS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    for column in CATEGORICAL_COLUMNS:
        prepared[column] = prepared[column].astype(str).str.strip().replace("", "Unknown")

    return prepared


def build_decision_tree_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, NUMERIC_COLUMNS),
            ("categorical", categorical_transformer, CATEGORICAL_COLUMNS),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=42)),
        ]
    )


def train_and_save_model() -> Any:
    if not TRAINING_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training file '{TRAINING_DATA_PATH}' was not found, and no saved model exists."
        )

    training_data = normalize_columns(pd.read_csv(TRAINING_DATA_PATH))
    missing_columns = [
        column for column in [*REQUIRED_COLUMNS, TARGET_COLUMN] if column not in training_data.columns
    ]
    if missing_columns:
        raise ValueError("Training data is missing required columns: " + ", ".join(missing_columns))

    features = training_data[REQUIRED_COLUMNS].copy()
    features = features.dropna(how="all")

    target = training_data.loc[features.index, TARGET_COLUMN]
    missing_target = target.isna() | (target.astype(str).str.strip() == "")
    if missing_target.any():
        rows = [str(index + 2) for index in target.index[missing_target].tolist()]
        raise ValueError(f"Training target '{TARGET_COLUMN}' is missing in CSV row(s): {', '.join(rows)}")

    features = features.loc[~missing_target].copy()
    for column in NUMERIC_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")
    for column in CATEGORICAL_COLUMNS:
        features[column] = features[column].map(lambda value: value.strip() if isinstance(value, str) else value)
        features[column] = features[column].replace("", np.nan)

    labels = target.loc[~missing_target]

    model = build_decision_tree_pipeline()
    model.fit(features, labels)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    return model


@st.cache_resource(show_spinner=False)
def ensure_model() -> tuple[Any, str]:
    if not MODEL_PATH.exists():
        model = train_and_save_model()
        return model, f"trained a new decision tree from {TRAINING_DATA_PATH} and saved it to {MODEL_PATH}"

    return joblib.load(MODEL_PATH), f"existing trained model from {MODEL_PATH}"


def get_feature_names(model: Any) -> list[str] | None:
    if hasattr(model, "feature_names_in_"):
        return [str(feature) for feature in model.feature_names_in_]

    pipeline_steps = getattr(model, "steps", None)
    if pipeline_steps:
        final_estimator = pipeline_steps[-1][1]
        if hasattr(final_estimator, "feature_names_in_"):
            return [str(feature) for feature in final_estimator.feature_names_in_]

    return None


def predict_with_trained_model(model: Any, prepared: pd.DataFrame) -> tuple[pd.Series, pd.Series | None, str]:
    try:
        predictions = model.predict(prepared)
        probabilities = predict_probability(model, prepared)
        return pd.Series(predictions, index=prepared.index), probabilities, "raw columns"
    except Exception as raw_error:
        encoded = pd.get_dummies(prepared, columns=CATEGORICAL_COLUMNS)
        feature_names = get_feature_names(model)
        if feature_names:
            encoded = encoded.reindex(columns=feature_names, fill_value=0)

        try:
            predictions = model.predict(encoded)
            probabilities = predict_probability(model, encoded)
            return pd.Series(predictions, index=prepared.index), probabilities, "one-hot encoded columns"
        except Exception as encoded_error:
            raise RuntimeError(
                "The model could not classify this file. "
                "Use a scikit-learn Pipeline that accepts the required Excel columns, "
                "or make sure the decision tree has feature_names_in_ matching the encoded training columns. "
                f"Raw prediction error: {raw_error}. Encoded prediction error: {encoded_error}."
            ) from encoded_error


def predict_probability(model: Any, features: pd.DataFrame) -> pd.Series | None:
    if not hasattr(model, "predict_proba"):
        return None

    probabilities = model.predict_proba(features)
    if probabilities.ndim != 2 or probabilities.shape[1] == 0:
        return None

    return pd.Series(probabilities.max(axis=1), index=features.index)


def demo_decision_tree_classifier(prepared: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    score = pd.Series(0.0, index=prepared.index)

    score += prepared["Lifespan"].clip(lower=0, upper=50) / 50 * 35
    score += (prepared["% Primary Material"].clip(lower=0, upper=100) / 100) * 20
    score += (prepared["% Secondary Material"].clip(lower=0, upper=100) > 0).astype(float) * 10

    easy_connections = prepared["Connection Method"].str.lower().str.contains(
        "bolt|screw|clip|mechanical|dry|demount|removable",
        regex=True,
        na=False,
    )
    hard_connections = prepared["Connection Method"].str.lower().str.contains(
        "weld|glue|adhesive|cast|chemical|permanent",
        regex=True,
        na=False,
    )
    score += easy_connections.astype(float) * 25
    score -= hard_connections.astype(float) * 20
    score += prepared["Weight"].clip(lower=0, upper=500) / 500 * 10

    labels = pd.cut(
        score,
        bins=[-float("inf"), 39, 69, float("inf")],
        labels=["Low reuse potential", "Needs review", "High reuse potential"],
    ).astype(str)
    confidence = ((score - 50).abs() / 50).clip(lower=0.35, upper=0.95)

    return labels, confidence


def classify_data(validated: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    prepared = prepare_data_for_model(validated)
    output = validated.copy()
    model, model_note = ensure_model()

    predictions, probabilities, strategy = predict_with_trained_model(model, prepared)
    output[PREDICTION_COLUMN] = predictions
    if probabilities is not None:
        output[CONFIDENCE_COLUMN] = probabilities
    return output, f"{model_note}; prediction used {strategy}"


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="classification")
    return buffer.getvalue()


def show_validation_errors(errors: list[str]) -> None:
    st.error("Data validation failed.")
    for error in errors:
        st.write(f"- {error}")


def render_upload_step() -> pd.DataFrame | None:
    st.subheader("1. Upload Excel File")
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=["xlsx", "xls"],
        help="The first sheet must contain the required component columns.",
    )

    if uploaded_file is None:
        st.info("Upload an Excel file to start.")
        return None

    try:
        data = read_excel(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the Excel file: {exc}")
        return None

    st.success(f"Loaded {len(data):,} row(s) and {len(data.columns):,} column(s).")
    with st.expander("Preview uploaded data", expanded=True):
        st.dataframe(data.head(20), use_container_width=True)
    return data


def render_validation_and_model_step(data: pd.DataFrame) -> pd.DataFrame | None:
    st.subheader("2. Validate Data and Run Model")
    validation = validate_data(data)

    if validation.warnings:
        for warning in validation.warnings:
            st.warning(warning)

    if not validation.is_valid:
        show_validation_errors(validation.errors)
        return None

    st.success("Validation passed. The data is ready for the model.")
    left, right = st.columns([1, 2])
    with left:
        st.metric("Rows ready", f"{len(validation.cleaned_data):,}")
        st.metric("Required columns", f"{len(REQUIRED_COLUMNS):,}")
    with right:
        st.dataframe(validation.cleaned_data.head(20), use_container_width=True)

    if st.button("Classify Excel Data", type="primary", use_container_width=True):
        try:
            classified, model_note = classify_data(validation.cleaned_data)
        except Exception as exc:
            st.error(str(exc))
            return None

        st.session_state["classified_data"] = classified
        st.session_state["model_note"] = model_note

    classified_data = st.session_state.get("classified_data")
    if classified_data is not None:
        st.success(f"Classification complete with {st.session_state.get('model_note')}.")
        return classified_data

    return None


def render_dashboard(classified: pd.DataFrame) -> None:
    st.subheader("3. Results Dashboard")

    total_components = len(classified)
    total_weight = classified["Weight"].sum()
    unique_components = classified["Component"].nunique()
    average_lifespan = classified["Lifespan"].mean()

    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows classified", f"{total_components:,}")
    metric_columns[1].metric("Total weight", f"{total_weight:,.2f}")
    metric_columns[2].metric("Component types", f"{unique_components:,}")
    metric_columns[3].metric("Avg lifespan", f"{average_lifespan:,.1f}")

    chart_left, chart_right = st.columns(2)
    with chart_left:
        class_counts = classified[PREDICTION_COLUMN].value_counts().reset_index()
        class_counts.columns = [PREDICTION_COLUMN, "Count"]
        st.plotly_chart(
            px.bar(
                class_counts,
                x=PREDICTION_COLUMN,
                y="Count",
                color=PREDICTION_COLUMN,
                title="Classification Counts",
            ),
            use_container_width=True,
        )

    with chart_right:
        material_weight = (
            classified.groupby("Primary Material", as_index=False)["Weight"]
            .sum()
            .sort_values("Weight", ascending=False)
        )
        st.plotly_chart(
            px.pie(
                material_weight,
                names="Primary Material",
                values="Weight",
                title="Weight by Primary Material",
                hole=0.35,
            ),
            use_container_width=True,
        )

    chart_bottom_left, chart_bottom_right = st.columns(2)
    with chart_bottom_left:
        st.plotly_chart(
            px.scatter(
                classified,
                x="Weight",
                y="Lifespan",
                color=PREDICTION_COLUMN,
                hover_data=["Component", "Sub-component", "Primary Material", "Connection Method"],
                title="Weight vs Lifespan",
            ),
            use_container_width=True,
        )

    with chart_bottom_right:
        component_summary = (
            classified.groupby(["Component", PREDICTION_COLUMN], as_index=False)
            .agg(Count=("Component", "size"), Weight=("Weight", "sum"))
            .sort_values(["Component", "Weight"], ascending=[True, False])
        )
        st.plotly_chart(
            px.bar(
                component_summary,
                x="Component",
                y="Weight",
                color=PREDICTION_COLUMN,
                title="Classification Weight by Component",
            ),
            use_container_width=True,
        )

    st.dataframe(classified, use_container_width=True)
    st.download_button(
        "Download classified Excel",
        data=dataframe_to_excel_bytes(classified),
        file_name="classified_components.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )


def render_required_columns() -> None:
    with st.expander("Required Excel columns"):
        st.write(pd.DataFrame({"Column name": REQUIRED_COLUMNS}))


def main() -> None:
    configure_page()
    st.title("Component Classification")
    st.caption("Upload component data, validate it, classify it with your decision tree, and review the dashboard.")

    try:
        with st.spinner("Checking trained decision tree model..."):
            _, model_note = ensure_model()
    except Exception as exc:
        st.error(f"Could not prepare the decision tree model: {exc}")
        st.stop()

    st.info(f"Model ready: {model_note}.")

    render_required_columns()

    uploaded_data = render_upload_step()
    if uploaded_data is None:
        return

    classified = render_validation_and_model_step(uploaded_data)
    if classified is None:
        return

    render_dashboard(classified)


if __name__ == "__main__":
    main()
