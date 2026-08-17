import io
import os
import re
from flask import Flask, Response, jsonify, render_template, request, session
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

from models.preprocessing import (
    clean_text_column,
    encode_column,
    find_and_replace,
    get_dataset_info,
    handle_missing_values,
    handle_outliers_iqr,
    inspect_uniqueness,
    modify_column_type,
    scale_feature,
)
from models.recommender import (
    check_recommendation_suitability,
    generate_recommendations,
)
from models.rfm import compute_rfm_segments
from models.sentiment import analyze_review_sentiments, check_review_suitability
from models.suitability import detect_segmentation_columns

app = Flask(__name__)
app.secret_key = "super_secret_ecom_key"
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global in-memory buffer to hold active working datasets and undo histories
# Key: filepath -> Value: list of DataFrames [df_original, df_edit_1, df_edit_2, ...]
DATASET_STACKS = {}


def read_csv_safely(filepath):
    """Attempts to read a CSV dataset across multiple encodings to prevent
    UnicodeDecodeError crashes on non-standard/Excel CSV files.
    """
    encodings_to_try = ["utf-8", "latin1", "iso-8859-1", "cp1252"]

    for enc in encodings_to_try:
        try:
            return pd.read_csv(filepath, encoding=enc)
        except (UnicodeDecodeError, Exception):
            continue

    # Fallback mode: ignore/replace unreadable characters
    return pd.read_csv(filepath, encoding="utf-8", errors="replace")


def get_current_df(filepath):
    """Retrieves the latest working copy of the dataset from memory stack, or loads from disk."""
    if filepath not in DATASET_STACKS or not DATASET_STACKS[filepath]:
        df = read_csv_safely(filepath)
        DATASET_STACKS[filepath] = [df]  # Base state at index 0
    return DATASET_STACKS[filepath][-1]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("dataset")
    if not file or file.filename == "":
        return "Please upload a valid CSV file.", 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Clean up old session files from memory buffer to prevent memory growth
    old_filepath = session.get("filepath")
    if old_filepath in DATASET_STACKS:
        DATASET_STACKS.pop(old_filepath, None)

    # Store new file state
    session["filepath"] = filepath
    session["filename"] = filename

    # Safely load the dataset regardless of character encoding
    try:
        df = read_csv_safely(filepath)
    except Exception as e:
        return f"Error reading dataset file: {str(e)}", 400

    # Initialize memory stack with freshly uploaded dataset
    DATASET_STACKS[filepath] = [df]

    # Clear lingering session flags
    session.pop("undo_available", None)
    session.pop("prep_history", None)

    suitability = detect_segmentation_columns(df)

    return render_template(
        "dashboard.html",
        filename=filename,
        suitability=suitability,
        preview=df.head(10).to_html(
            classes="table table-striped table-hover", index=False
        ),
        columns=df.columns.tolist(),
        total_rows=len(df),
        total_cols=len(df.columns),
    )


@app.route("/apply-preprocessing", methods=["POST"])
def apply_preprocessing():
    print("\n--- [CONSOLE DEBUG: /apply-preprocessing] ---")
    filepath = session.get("filepath")
    print(f"[DEBUG] Active session filepath: {filepath}")

    if not filepath or not os.path.exists(filepath):
        print("[DEBUG] ERROR: Active dataset not found or file path invalid.")
        return jsonify({"error": "No active dataset found"}), 400

    options = request.json or {}
    print(f"[DEBUG] Received preprocessing options: {options}")

    df = read_csv_safely(filepath)
    print(
        f"[DEBUG] Loaded dataset successfully. Initial shape: {df.shape if df is not None else 'None'}"
    )

    # Preprocessing
    missing_strat = options.get("missing_strategy", "drop")
    print(f"[DEBUG] Applying missing value strategy: '{missing_strat}'")

    for col in df.columns:
        print(f"[DEBUG] Processing missing values for column: '{col}'")
        df = handle_missing_values(df, col, missing_strat)

    # Save to disk AND update state
    df.to_csv(filepath, index=False, encoding="utf-8")
    DATASET_STACKS[filepath] = [df]
    print(f"[DEBUG] Updated dataset saved to disk and memory buffer. Final shape: {df.shape}")

    return jsonify(
        {
            "status": "success",
            "message": "Dataset updated successfully!",
            "preview": df.head(10).to_html(
                classes="table table-striped table-hover", index=False
            ),
        }
    )


@app.route("/run-segmentation", methods=["POST"])
def run_segmentation():
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "No dataset found in active session"}), 400

    data = request.json
    id_col = data.get("id_col")
    date_col = data.get("date_col")
    amount_col = data.get("amount_col")

    try:
        df = get_current_df(filepath)
        rfm_table = compute_rfm_segments(df, id_col, date_col, amount_col)

        # 1. HTML Table Preview
        html_output = rfm_table.head(15).to_html(
            classes="table table-striped table-hover border", index=False
        )

        # 2. Aggregations for Charts
        segment_counts = rfm_table["Segment"].value_counts().to_dict()
        segment_monetary = (
            rfm_table.groupby("Segment")["Monetary"].sum().round(2).to_dict()
        )

        return jsonify(
            {
                "status": "success",
                "html_table": html_output,
                "chart_data": {
                    "labels": list(segment_counts.keys()),
                    "counts": list(segment_counts.values()),
                    "monetary": [
                        segment_monetary.get(k, 0) for k in segment_counts.keys()
                    ],
                },
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/run-recommendations", methods=["POST"])
def run_recommendation_engine():
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "No dataset found in active session"}), 400

    data = request.json
    user_col = data.get("user_col")
    item_col = data.get("item_col")
    invoice_col = data.get("invoice_col")
    qty_col = data.get("qty_col")

    try:
        df = get_current_df(filepath)
        results = generate_recommendations(
            df=df,
            user_col=user_col,
            item_col=item_col,
            invoice_col=invoice_col,
            qty_col=qty_col,
            top_n=5,
        )
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/run-reviews", methods=["POST"])
def run_reviews():
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "No active dataset session"}), 400

    data = request.json or {}
    review_col = data.get("review_col")
    product_col = data.get("product_col")

    try:
        df = get_current_df(filepath)
        results = analyze_review_sentiments(df, review_col, product_col)
        return jsonify({"status": "success", "data": results, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- PREPROCESSING API ENDPOINTS ---

@app.route("/prep/info", methods=["GET"])
def prep_info():
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "No dataset session active"}), 400

    df = get_current_df(filepath)
    info = get_dataset_info(df)
    duplicates = int(df.duplicated().sum())

    return jsonify(
        {
            "status": "success",
            "info": info,
            "duplicates": duplicates,
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "columns": df.columns.tolist(),
        }
    )


@app.route("/prep/apply", methods=["POST"])
def prep_apply():
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "No dataset session active"}), 400

    data = request.json or {}
    action = data.get("action")
    col = data.get("column")

    current_df = get_current_df(filepath)

    try:
        # --- UNDO / SAVE ACTIONS ---
        if action == "undo":
            if filepath in DATASET_STACKS and len(DATASET_STACKS[filepath]) > 1:
                DATASET_STACKS[filepath].pop()
                df = DATASET_STACKS[filepath][-1]
                undo_count = len(DATASET_STACKS[filepath]) - 1
                msg = f"Last operation undone. ({undo_count} unsaved change(s) remaining)"
            else:
                return jsonify({"error": "Nothing to undo."}), 400

        elif action == "save":
            df = current_df
            df.to_csv(filepath, index=False, encoding="utf-8")
            DATASET_STACKS[filepath] = [df]
            msg = "All changes saved permanently to dataset!"

        # --- TRANSFORMATION ACTIONS ---
        else:
            df = current_df.copy()

            if action == "missing":
                strategy = data.get("strategy")
                df = handle_missing_values(df, col, strategy)

            elif action == "drop_duplicates":
                df = df.drop_duplicates()

            elif action == "type_modify":
                target_type = data.get("target_type")
                df = modify_column_type(df, col, target_type)

            elif action == "replace":
                find_val = data.get("find")
                replace_val = data.get("replace")

                if not col or col not in df.columns:
                    return jsonify(
                        {
                            "status": "error",
                            "message": "Invalid column selected.",
                        }
                    )

                if replace_val == "" or replace_val is None:
                    parsed_replace = np.nan
                else:
                    parsed_replace = replace_val

                dtype = df[col].dtype

                try:
                    if pd.api.types.is_integer_dtype(dtype):
                        parsed_find = int(find_val)
                        if not pd.isna(parsed_replace):
                            parsed_replace = int(replace_val)

                    elif pd.api.types.is_float_dtype(dtype):
                        parsed_find = float(find_val)
                        if not pd.isna(parsed_replace):
                            parsed_replace = float(replace_val)

                    elif pd.api.types.is_bool_dtype(dtype):
                        parsed_find = str(find_val).strip().lower() in [
                            "true",
                            "1",
                            "yes",
                        ]
                        if not pd.isna(parsed_replace):
                            parsed_replace = str(replace_val).strip().lower() in [
                                "true",
                                "1",
                                "yes",
                            ]

                    else:
                        parsed_find = str(find_val)
                        if not pd.isna(parsed_replace):
                            parsed_replace = str(replace_val)

                    df[col] = df[col].replace(parsed_find, parsed_replace)

                except ValueError:
                    return jsonify(
                        {
                            "status": "error",
                            "error": f"Cannot convert '{find_val}' or '{replace_val}' to column type ({dtype}).",
                        }
                    )

            elif action == "encode":
                method = data.get("method")
                df = encode_column(df, col, method)

            elif action == "outliers":
                df = handle_outliers_iqr(df, col)

            elif action == "scale":
                method = data.get("method")
                df = scale_feature(df, col, method)

            elif action == "clean_text":
                df = clean_text_column(df, col)

            else:
                return jsonify({"error": f"Unknown action '{action}'"}), 400

            DATASET_STACKS[filepath].append(df)
            undo_count = len(DATASET_STACKS[filepath]) - 1
            msg = f"Successfully executed '{action}' operation! ({undo_count} unsaved change(s))"

        preview_html = df.head(10).to_html(
            classes="table table-striped table-hover", index=False
        )
        undo_available = len(DATASET_STACKS.get(filepath, [])) > 1

        return jsonify(
            {
                "status": "success",
                "message": msg,
                "preview": preview_html,
                "undo_available": undo_available,
                "total_rows": len(df),
                "total_cols": len(df.columns),
                "duplicates": int(df.duplicated().sum()),
                "info": get_dataset_info(df),
                "columns": df.columns.tolist(),
            }
        )

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": f"Operation failed: {str(e)}"}), 500


@app.route("/prep/inspect-unique", methods=["POST"])
def prep_inspect_unique():
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "No dataset session active"}), 400

    col = request.json.get("column")
    df = get_current_df(filepath)
    res = inspect_uniqueness(df, col)

    return jsonify({"status": "success", "data": res})


@app.route("/prep/check-suitability", methods=["POST"])
def check_suitability():
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "No dataset session active"}), 400

    df = get_current_df(filepath)
    check_result = detect_segmentation_columns(df)

    if not check_result["is_suitable"]:
        return (
            jsonify(
                {
                    "status": "unsuitable",
                    "message": "Dataset is missing required features.",
                    "missing": check_result["missing_requirements"],
                }
            ),
            400,
        )

    return jsonify(
        {
            "status": "success",
            "all_columns": df.columns.tolist(),
            "detected_mapping": check_result["mapping"],
        }
    )


@app.route("/prep/derive-column", methods=["POST"])
def derive_column():
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"status": "error", "message": "No active dataset session"}), 400

    data = request.get_json() or {}
    new_col = data.get("col_name", "").strip()
    formula = data.get("formula", "").strip()

    if not new_col or not formula:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": "Column name and formula are required.",
                }
            ),
            400,
        )

    try:
        current_df = get_current_df(filepath)
        df = current_df.copy()

        # Evaluate expression using pandas eval
        df[new_col] = df.eval(formula)

        # Push modified DataFrame to memory stack (aligns with the undo history workflow)
        DATASET_STACKS[filepath].append(df)

        return jsonify(
            {
                "status": "success",
                "message": f"Column '{new_col}' successfully created!",
                "all_columns": list(df.columns),
            }
        )

    except Exception as e:
        return (
            jsonify(
                {
                    "status": "error",
                    "message": f"Formula Evaluation Error: {str(e)}",
                }
            ),
            400,
        )


@app.route("/prep/check-recommendation-suitability", methods=["POST"])
def check_rec_suitability():
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"status": "error", "message": "No active dataset"}), 400

    df = get_current_df(filepath)
    suitability_info = check_recommendation_suitability(df)

    return jsonify({
        "status": "success",
        "is_suitable": suitability_info["is_suitable"],
        "detected_mapping": suitability_info["detected_mapping"],
        "all_columns": df.columns.tolist()
    })


@app.route("/prep/check-review-suitability", methods=["POST"])
def check_review_suitability_route():
    print("\n--- [CONSOLE DEBUG: /prep/check-review-suitability] ---")
    filepath = session.get("filepath")
    print(f"[DEBUG] Active session filepath: {filepath}")

    if not filepath or not os.path.exists(filepath):
        print("[DEBUG] ERROR: Filepath is missing or file does not exist on disk.")
        return jsonify({"status": "error", "message": "No active dataset"}), 400

    df = get_current_df(filepath)
    print(f"[DEBUG] Dataset loaded into memory. Columns detected: {df.columns.tolist()}")

    print("[DEBUG] Running check_review_suitability logic...")
    suitability_info = check_review_suitability(df)
    print(f"[DEBUG] Raw suitability output: {suitability_info}")

    is_suitable = suitability_info.get("is_suitable", False)

    if not is_suitable:
        missing_reqs = suitability_info.get("missing_requirements", [])
        print(f"[DEBUG] UNSUITABILITY DETECTED! Missing features: {missing_reqs}")
        return jsonify(
            {
                "status": "unsuitable",
                "message": "Dataset is missing required review features.",
                "missing": missing_reqs,
            }
        ), 400

    mapping = suitability_info.get("mapping") or suitability_info.get(
        "detected_mapping", {}
    )
    print(f"[DEBUG] SUITABILITY CONFIRMED. Column mapping auto-detected: {mapping}")

    return jsonify(
        {
            "status": "success",
            "is_suitable": True,
            "all_columns": df.columns.tolist(),
            "detected_mapping": mapping,
        }
    )

@app.route("/prep/export-sentiment-csv", methods=["GET"])
def export_sentiment_csv():
    print("\n--- [CONSOLE DEBUG: /prep/export-sentiment-csv] ---")
    filepath = session.get("filepath")
    print(f"[DEBUG] Active session filepath: {filepath}")

    if not filepath or not os.path.exists(filepath):
        print("[DEBUG] ERROR: Export failed. Active dataset not found.")
        return "No active dataset found to export.", 400

    review_col = request.args.get("review_col")
    product_col = request.args.get("product_col")
    print(f"[DEBUG] Export parameters received -> Review Column: '{review_col}', Product Column: '{product_col}'")

    if not review_col:
        print("[DEBUG] ERROR: Required parameter 'review_col' is missing.")
        return "Review column parameters are missing.", 400

    try:
        df = get_current_df(filepath)
        print(f"[DEBUG] Analyzing sentiments for export. Row count: {len(df)}")
        sentiments = analyze_review_sentiments(df, review_col, product_col)

        if isinstance(sentiments, dict) and "processed_df" in sentiments:
            print("[DEBUG] Sentiment analysis returned dictionary containing 'processed_df'.")
            export_df = pd.DataFrame(sentiments["processed_df"])
        elif isinstance(sentiments, pd.DataFrame):
            print("[DEBUG] Sentiment analysis returned direct DataFrame.")
            export_df = sentiments
        else:
            print("[DEBUG] WARNING: Sentiment analysis returned unexpected format. Fallback to original DataFrame.")
            export_df = df.copy()

        print(f"[DEBUG] Export DataFrame constructed successfully. Output shape: {export_df.shape}")

        output_stream = io.StringIO()
        export_df.to_csv(output_stream, index=False)

        print("[DEBUG] Successfully compiled CSV stream. Sending response attachment.")
        return Response(
            output_stream.getvalue().encode("utf-8"),
            mimetype="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": "attachment; filename=sentiment_analysis_results.csv"
            },
        )
    except Exception as e:
        print(f"[DEBUG] CRITICAL ERROR during CSV export: {str(e)}")
        return f"Error exporting CSV: {str(e)}", 500
        
if __name__ == "__main__":
    app.run(debug=True)