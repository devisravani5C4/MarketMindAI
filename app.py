import csv
import io
import json
import os
import re
import uuid
from datetime import datetime
from functools import wraps

from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename

from models.database import Dataset, User, db
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
    predict_likely_buyers,
)
from models.rfm import compute_rfm_segments
from models.sentiment import analyze_review_sentiments, check_review_suitability
from models.suitability import detect_segmentation_columns
from models.code_console import drop_kernel, execute_code, reset_kernel

app = Flask(__name__)
app.secret_key = "super_secret_ecom_key"
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- Database configuration (SQLite -- no external DB server required) ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + os.path.join(BASE_DIR, "marketmind.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

with app.app_context():
    db.create_all()

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global in-memory buffer to hold active working datasets and undo histories
# Key: filepath -> Value: list of DataFrames [df_original, df_edit_1, df_edit_2, ...]
DATASET_STACKS = {}


def login_required(view_func):
    """Redirects to /login if there's no authenticated user in the session.
    Applied to every route that touches a dataset or user-specific data."""

    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped_view


def cache_module_result(field_name, payload):
    """Persists the JSON result of an analysis module against the currently
    active Dataset row (if any) so it can be restored later from history."""
    dataset_id = session.get("dataset_id")
    if not dataset_id:
        return
    dataset_record = Dataset.query.get(dataset_id)
    if not dataset_record:
        return
    setattr(dataset_record, field_name, json.dumps(payload))
    db.session.commit()


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


def convert_file_to_dataframe(filepath, original_filename):
    """
    Reads a non-CSV dataset file (Excel, JSON, TSV, or plain delimited text)
    into a pandas DataFrame entirely offline using pandas/openpyxl — no
    external conversion services involved, so it keeps working even with no
    internet connection (handy at an expo booth). Raises ValueError with a
    friendly message on unsupported or unreadable files.
    """
    ext = os.path.splitext(original_filename)[1].lower()

    try:
        if ext in (".xlsx", ".xlsm"):
            return pd.read_excel(filepath, engine="openpyxl")
        elif ext == ".xls":
            # Legacy Excel format; requires the optional 'xlrd' package.
            return pd.read_excel(filepath)
        elif ext == ".json":
            return pd.read_json(filepath)
        elif ext == ".tsv":
            return pd.read_csv(filepath, sep="\t")
        elif ext == ".txt":
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                sample = f.read(4096)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
                sep = dialect.delimiter
            except csv.Error:
                sep = ","
            return pd.read_csv(filepath, sep=sep, engine="python")
        elif ext == ".csv":
            return read_csv_safely(filepath)
        else:
            raise ValueError(
                f"'{ext or 'this file type'}' isn't supported yet. "
                "Try .xlsx, .xls, .json, .tsv, .txt, or .csv."
            )
    except ValueError:
        raise
    except ImportError as ie:
        raise ValueError(
            f"Reading this file needs an extra library ('{ie.name}') that isn't installed here. "
            "Try re-saving it as .xlsx instead."
        )
    except Exception as e:
        raise ValueError(f"Couldn't read this file: {str(e)}")


def get_current_df(filepath):
    """Retrieves the latest working copy of the dataset from memory stack, or loads from disk."""
    if filepath not in DATASET_STACKS or not DATASET_STACKS[filepath]:
        df = read_csv_safely(filepath)
        DATASET_STACKS[filepath] = [df]  # Base state at index 0
    return DATASET_STACKS[filepath][-1]


# --- AUTHENTICATION ---

@app.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("home"))

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        confirm_password = request.form.get("confirm_password") or ""

        if not username or not password:
            return render_template("register.html", error="Username and password are required.")
        if len(password) < 6:
            return render_template("register.html", error="Password must be at least 6 characters.")
        if password != confirm_password:
            return render_template("register.html", error="Passwords do not match.")
        if User.query.filter_by(username=username).first():
            return render_template("register.html", error="That username is already taken.")

        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        session["user_id"] = user.id
        session["username"] = user.username
        return redirect(url_for("home"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("home"))

    if request.method == "POST":
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""

        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            return render_template("login.html", error="Invalid username or password.")

        session["user_id"] = user.id
        session["username"] = user.username
        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# --- HOME / UPLOAD / HISTORY ---

@app.route("/")
@login_required
def home():
    user = User.query.get(session["user_id"])
    datasets = [d.to_dict() for d in user.datasets]
    return render_template("index.html", datasets=datasets, username=user.username)


@app.route("/upload", methods=["POST"])
@login_required
def upload():
    file = request.files.get("dataset")
    if not file or file.filename == "":
        return "Please upload a valid CSV file.", 400

    user_id = session["user_id"]

    # Each user gets their own upload subfolder so two users can't collide
    # on filename or silently overwrite each other's files.
    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], str(user_id))
    os.makedirs(user_folder, exist_ok=True)

    original_name = secure_filename(file.filename)
    stored_filename = f"{uuid.uuid4().hex}_{original_name}"
    filepath = os.path.join(user_folder, stored_filename)
    file.save(filepath)

    # Clean up old session files from memory buffer to prevent memory growth
    old_filepath = session.get("filepath")
    if old_filepath in DATASET_STACKS:
        DATASET_STACKS.pop(old_filepath, None)
        drop_kernel(old_filepath)

    # Safely load the dataset regardless of character encoding
    try:
        df = read_csv_safely(filepath)
    except Exception as e:
        return f"Error reading dataset file: {str(e)}", 400

    # Record this upload in the user's dataset history
    dataset_record = Dataset(
        user_id=user_id,
        display_name=original_name,
        stored_filename=stored_filename,
        filepath=filepath,
        total_rows=len(df),
        total_cols=len(df.columns),
    )
    db.session.add(dataset_record)
    db.session.commit()

    # Store new file state
    session["filepath"] = filepath
    session["filename"] = original_name
    session["dataset_id"] = dataset_record.id

    # Initialize memory stack with freshly uploaded dataset
    DATASET_STACKS[filepath] = [df]

    # Clear lingering session flags
    session.pop("undo_available", None)
    session.pop("prep_history", None)

    suitability = detect_segmentation_columns(df)

    return render_template(
        "dashboard.html",
        filename=original_name,
        suitability=suitability,
        preview=df.head(10).to_html(
            classes="table table-striped table-hover", index=False
        ),
        columns=df.columns.tolist(),
        total_rows=len(df),
        total_cols=len(df.columns),
        cached_results={},
    )


@app.route("/convert-to-csv", methods=["POST"])
@login_required
def convert_to_csv():
    file = request.files.get("raw_file")
    if not file or file.filename == "":
        user = User.query.get(session["user_id"])
        datasets = [d.to_dict() for d in user.datasets]
        return render_template(
            "index.html", datasets=datasets, username=user.username,
            convert_error="Please choose a file to convert.",
        )

    user_id = session["user_id"]
    original_name = secure_filename(file.filename)

    # Save the raw upload to a temp path first so pandas/openpyxl can read it.
    temp_folder = os.path.join(app.config["UPLOAD_FOLDER"], str(user_id), "_tmp_convert")
    os.makedirs(temp_folder, exist_ok=True)
    temp_path = os.path.join(temp_folder, f"{uuid.uuid4().hex}_{original_name}")
    file.save(temp_path)

    try:
        df = convert_file_to_dataframe(temp_path, original_name)
        if df is None or df.empty or len(df.columns) == 0:
            raise ValueError("That file converted to an empty table — please check it actually has data in it.")
    except ValueError as ve:
        user = User.query.get(session["user_id"])
        datasets = [d.to_dict() for d in user.datasets]
        return render_template(
            "index.html", datasets=datasets, username=user.username, convert_error=str(ve),
        )
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

    # From here it's identical to a normal CSV upload: save the converted
    # table as a real .csv file and jump straight into the dashboard.
    display_name = os.path.splitext(original_name)[0] + ".csv"
    user_folder = os.path.join(app.config["UPLOAD_FOLDER"], str(user_id))
    os.makedirs(user_folder, exist_ok=True)
    stored_filename = f"{uuid.uuid4().hex}_{display_name}"
    filepath = os.path.join(user_folder, stored_filename)
    df.to_csv(filepath, index=False, encoding="utf-8")

    old_filepath = session.get("filepath")
    if old_filepath in DATASET_STACKS:
        DATASET_STACKS.pop(old_filepath, None)
        drop_kernel(old_filepath)

    dataset_record = Dataset(
        user_id=user_id,
        display_name=display_name,
        stored_filename=stored_filename,
        filepath=filepath,
        total_rows=len(df),
        total_cols=len(df.columns),
    )
    db.session.add(dataset_record)
    db.session.commit()

    session["filepath"] = filepath
    session["filename"] = display_name
    session["dataset_id"] = dataset_record.id
    DATASET_STACKS[filepath] = [df]
    session.pop("undo_available", None)
    session.pop("prep_history", None)

    suitability = detect_segmentation_columns(df)

    return render_template(
        "dashboard.html",
        filename=display_name,
        suitability=suitability,
        preview=df.head(10).to_html(
            classes="table table-striped table-hover", index=False
        ),
        columns=df.columns.tolist(),
        total_rows=len(df),
        total_cols=len(df.columns),
        cached_results={},
        converted_from=original_name,
    )


@app.route("/history/load/<int:dataset_id>")
@login_required
def load_history_dataset(dataset_id):
    dataset_record = Dataset.query.filter_by(
        id=dataset_id, user_id=session["user_id"]
    ).first()

    if not dataset_record or not os.path.exists(dataset_record.filepath):
        return "That dataset could not be found. It may have been deleted.", 404

    filepath = dataset_record.filepath
    df = read_csv_safely(filepath)
    DATASET_STACKS[filepath] = [df]
    drop_kernel(filepath)  # force a fresh Code-tab kernel for this (re)loaded dataset

    session["filepath"] = filepath
    session["filename"] = dataset_record.display_name
    session["dataset_id"] = dataset_record.id
    session.pop("undo_available", None)
    session.pop("prep_history", None)

    dataset_record.last_opened_at = datetime.utcnow()
    db.session.commit()

    suitability = detect_segmentation_columns(df)

    # Restore the last saved analysis (if any) for each module so the
    # dashboard can render it immediately instead of starting from scratch.
    cached_results = {
        "segmentation": json.loads(dataset_record.last_segmentation_result)
        if dataset_record.last_segmentation_result else None,
        "recommendation": json.loads(dataset_record.last_recommendation_result)
        if dataset_record.last_recommendation_result else None,
        "reviews": json.loads(dataset_record.last_review_result)
        if dataset_record.last_review_result else None,
    }

    # Jump straight back to the furthest module the user had already worked
    # on, instead of always dropping them back on the Overview tab.
    if cached_results["reviews"]:
        active_section = "reviews"
    elif cached_results["recommendation"]:
        active_section = "recommendation"
    elif cached_results["segmentation"]:
        active_section = "clustering"
    else:
        active_section = "overview"

    return render_template(
        "dashboard.html",
        filename=dataset_record.display_name,
        suitability=suitability,
        preview=df.head(10).to_html(
            classes="table table-striped table-hover", index=False
        ),
        columns=df.columns.tolist(),
        total_rows=len(df),
        total_cols=len(df.columns),
        cached_results=cached_results,
        active_section=active_section,
    )


@app.route("/history/delete/<int:dataset_id>", methods=["POST"])
@login_required
def delete_history_dataset(dataset_id):
    dataset_record = Dataset.query.filter_by(
        id=dataset_id, user_id=session["user_id"]
    ).first()

    if not dataset_record:
        return jsonify({"error": "Dataset not found"}), 404

    DATASET_STACKS.pop(dataset_record.filepath, None)
    drop_kernel(dataset_record.filepath)
    if os.path.exists(dataset_record.filepath):
        try:
            os.remove(dataset_record.filepath)
        except OSError:
            pass

    if session.get("dataset_id") == dataset_record.id:
        session.pop("filepath", None)
        session.pop("filename", None)
        session.pop("dataset_id", None)

    db.session.delete(dataset_record)
    db.session.commit()

    return jsonify({"status": "success"})


@app.route("/apply-preprocessing", methods=["POST"])
@login_required
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
@login_required
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

        response_payload = {
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
        cache_module_result("last_segmentation_result", response_payload)
        return jsonify(response_payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/run-recommendations", methods=["POST"])
@login_required
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
        cache_module_result("last_recommendation_result", results)
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/run-new-product-prediction", methods=["POST"])
@login_required
def run_new_product_prediction():
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "No dataset found in active session"}), 400

    data = request.json or {}
    product_name = (data.get("product_name") or "").strip()
    user_col = data.get("user_col")
    item_col = data.get("item_col")
    qty_col = data.get("qty_col")

    if not product_name:
        return jsonify({"error": "Please enter a product name."}), 400
    if not user_col or not item_col:
        return jsonify({"error": "Select a Customer ID and Product column above first."}), 400

    try:
        df = get_current_df(filepath)
        results = predict_likely_buyers(df, user_col, item_col, product_name, qty_col)
        results["product_name"] = product_name
        return jsonify({"status": "success", "results": results})
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/run-reviews", methods=["POST"])
@login_required
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
        cache_module_result("last_review_result", results)
        return jsonify({"status": "success", "data": results, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------------------------------------------------------
# Customer-facing "storefront" demo — a friendly, non-technical view that
# shows the *effect* of the three analytics engines above (segmentation,
# recommendations, sentiment) as an actual shopper would experience them.
# It only reads data already cached against the active dataset; it never
# recomputes anything itself (aside from the tiny live-sentiment textbox).
# ---------------------------------------------------------------------------
PERSONA_COPY = {
    "VIP / Whale": {
        "emoji": "🐋",
        "greeting": "Welcome back, valued VIP!",
        "message": "You're one of our top spenders — enjoy first dibs on new arrivals and a dedicated support line.",
        "offer": "VIP-only: free priority shipping on every order",
        "theme": "linear-gradient(135deg,#7c3aed,#3b82f6)",
    },
    "Premium / Champion": {
        "emoji": "🏆",
        "greeting": "Hey Champion, great to see you!",
        "message": "You shop often and shop big — here's early access to today's flash deals.",
        "offer": "Champion perk: extra 15% off, ends tonight",
        "theme": "linear-gradient(135deg,#f59e0b,#ec4899)",
    },
    "Loyal Customer": {
        "emoji": "💛",
        "greeting": "Welcome back, loyal friend!",
        "message": "Thanks for coming back again and again — here's something to say thanks.",
        "offer": "Loyalty reward: 10% off your next order",
        "theme": "linear-gradient(135deg,#22c55e,#14b8a6)",
    },
    "Discount Seeker": {
        "emoji": "🏷️",
        "greeting": "Deal alert, just for you!",
        "message": "We know you love a good bargain — check out today's clearance picks.",
        "offer": "Today only: extra 20% off clearance",
        "theme": "linear-gradient(135deg,#ef4444,#f59e0b)",
    },
    "At Risk": {
        "emoji": "💌",
        "greeting": "We miss you!",
        "message": "It's been a while — come back and see what's new.",
        "offer": "Welcome-back offer: 20% off to come back",
        "theme": "linear-gradient(135deg,#3b82f6,#7c3aed)",
    },
    "Lost Customer": {
        "emoji": "🔔",
        "greeting": "Long time no see!",
        "message": "A lot has changed since your last visit — here's a big reason to give us another try.",
        "offer": "One-time comeback deal: 25% off storewide",
        "theme": "linear-gradient(135deg,#64748b,#334155)",
    },
    "Standard Customer": {
        "emoji": "🙂",
        "greeting": "Welcome!",
        "message": "Explore today's picks curated just for shoppers like you.",
        "offer": "New here? Get 10% off your first order",
        "theme": "linear-gradient(135deg,#14b8a6,#3b82f6)",
    },
}
DEFAULT_PERSONA = {
    "emoji": "🛍️",
    "greeting": "Welcome, shopper!",
    "message": "Explore today's picks curated just for you.",
    "offer": "New here? Get 10% off your first order",
    "theme": "linear-gradient(135deg,#7c3aed,#ec4899)",
}


@app.route("/storefront")
@login_required
def storefront():
    dataset_id = session.get("dataset_id")
    dataset_record = Dataset.query.get(dataset_id) if dataset_id else None

    segmentation = (
        json.loads(dataset_record.last_segmentation_result)
        if dataset_record and dataset_record.last_segmentation_result else None
    )
    recommendation = (
        json.loads(dataset_record.last_recommendation_result)
        if dataset_record and dataset_record.last_recommendation_result else None
    )
    reviews = (
        json.loads(dataset_record.last_review_result)
        if dataset_record and dataset_record.last_review_result else None
    )

    # Personas to switch between: real segment names if segmentation has been
    # run, otherwise a friendly generic demo set so the page never feels empty.
    if segmentation and segmentation.get("chart_data", {}).get("labels"):
        persona_names = segmentation["chart_data"]["labels"]
    else:
        persona_names = ["Premium / Champion", "Loyal Customer", "At Risk", "Standard Customer"]

    personas = [
        {"name": name, **PERSONA_COPY.get(name, DEFAULT_PERSONA)}
        for name in persona_names
    ]

    # Build a small product catalog from whatever product names show up across
    # the recommendation + review outputs for this dataset.
    product_names = []
    if recommendation and recommendation.get("popular_items"):
        product_names += [p["item"] for p in recommendation["popular_items"]]
    if reviews and reviews.get("product_breakdown"):
        product_names += [p["product_name"] for p in reviews["product_breakdown"]]

    seen = set()
    catalog = []
    for name in product_names:
        if name and name not in seen:
            seen.add(name)
            catalog.append(name)
    catalog = catalog[:8]

    sentiment_by_product = {}
    if reviews and reviews.get("product_breakdown"):
        for row in reviews["product_breakdown"]:
            sentiment_by_product[row["product_name"]] = row

    return render_template(
        "storefront.html",
        has_data=bool(dataset_record),
        filename=dataset_record.display_name if dataset_record else None,
        personas=personas,
        catalog=catalog,
        recommendation=recommendation,
        reviews=reviews,
        sentiment_by_product=sentiment_by_product,
    )


# --- PREPROCESSING API ENDPOINTS ---

@app.route("/prep/info", methods=["GET"])
@login_required
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


@app.route("/prep/download-current", methods=["GET"])
@login_required
def download_current_dataset():
    """Streams the current in-memory working copy of the active dataset
    (i.e. including every preprocessing step applied so far, even ones not
    yet saved back to disk) as a downloadable CSV."""
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return "No active dataset found to download.", 400

    df = get_current_df(filepath)

    base_name = session.get("filename", "dataset.csv")
    name_root, _, ext = base_name.rpartition(".")
    name_root = name_root or base_name
    download_name = f"{name_root}_preprocessed.csv"

    output_stream = io.StringIO()
    df.to_csv(output_stream, index=False)

    return Response(
        output_stream.getvalue().encode("utf-8"),
        mimetype="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{download_name}"'
        },
    )


@app.route("/prep/apply", methods=["POST"])
@login_required
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
@login_required
def prep_inspect_unique():
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"error": "No dataset session active"}), 400

    col = request.json.get("column")
    df = get_current_df(filepath)
    res = inspect_uniqueness(df, col)

    return jsonify({"status": "success", "data": res})


@app.route("/prep/check-suitability", methods=["POST"])
@login_required
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
@login_required
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
@login_required
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
@login_required
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
@login_required
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


# --- CODE TAB (Colab-style notebook) API ENDPOINTS ---
#
# These back the "Code" toggle in the dashboard: a persistent, per-dataset
# Python kernel that starts pre-loaded with the active dataset (`df`) --
# including any preprocessing already applied -- plus pandas/numpy/
# matplotlib. Each cell run keeps whatever state previous cells created,
# just like a real notebook kernel.

@app.route("/code/init", methods=["GET"])
@login_required
def code_init():
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"status": "error", "message": "No active dataset session"}), 400

    df = get_current_df(filepath)
    reset_kernel(filepath, lambda: get_current_df(filepath))
    prep_steps_pending = len(DATASET_STACKS.get(filepath, [])) - 1

    return jsonify(
        {
            "status": "success",
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "columns": df.columns.tolist(),
            "preview": df.head(5).to_html(
                classes="table table-sm table-striped table-hover mb-0", index=False
            ),
            "prep_steps_pending": max(prep_steps_pending, 0),
        }
    )


@app.route("/code/execute", methods=["POST"])
@login_required
def code_execute():
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"status": "error", "message": "No active dataset session"}), 400

    data = request.json or {}
    code = data.get("code", "")
    if not code.strip():
        return jsonify({"status": "error", "message": "Nothing to run."}), 400

    result = execute_code(filepath, code, lambda: get_current_df(filepath))
    result["status"] = "success"
    return jsonify(result)


@app.route("/code/reset", methods=["POST"])
@login_required
def code_reset():
    filepath = session.get("filepath")
    if not filepath or not os.path.exists(filepath):
        return jsonify({"status": "error", "message": "No active dataset session"}), 400

    df = get_current_df(filepath)
    reset_kernel(filepath, lambda: df)

    return jsonify(
        {
            "status": "success",
            "message": "Kernel restarted — dataset re-imported.",
            "total_rows": len(df),
            "total_cols": len(df.columns),
            "columns": df.columns.tolist(),
            "preview": df.head(5).to_html(
                classes="table table-sm table-striped table-hover mb-0", index=False
            ),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)