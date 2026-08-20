"""Database models for authentication and per-user dataset history.

Uses a lightweight SQLite database via Flask-SQLAlchemy -- no external
database server needs to be installed or configured. The .db file is
created automatically (in the project root) the first time the app runs.
"""
from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    datasets = db.relationship(
        "Dataset",
        backref="owner",
        lazy=True,
        cascade="all, delete-orphan",
        order_by="desc(Dataset.last_opened_at)",
    )

    def set_password(self, raw_password):
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password):
        return check_password_hash(self.password_hash, raw_password)


class Dataset(db.Model):
    """One row per uploaded dataset for a user.

    This is what lets the app show "past uploads" and let a user reload one
    instead of only ever tracking a single active dataset in the Flask
    session. It also caches the JSON result of the last successful run of
    each analysis module, so reloading a dataset can optionally restore the
    last analysis instead of forcing a full recompute.
    """

    __tablename__ = "datasets"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    display_name = db.Column(db.String(255), nullable=False)      # original filename shown to the user
    stored_filename = db.Column(db.String(255), nullable=False)   # unique name actually used on disk
    filepath = db.Column(db.String(500), nullable=False)

    total_rows = db.Column(db.Integer, default=0)
    total_cols = db.Column(db.Integer, default=0)

    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_opened_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Cached JSON blobs (as text) of the last successful result from each
    # module. Nullable -- a freshly uploaded dataset has none of these yet.
    last_segmentation_result = db.Column(db.Text, nullable=True)
    last_recommendation_result = db.Column(db.Text, nullable=True)
    last_review_result = db.Column(db.Text, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "display_name": self.display_name,
            "total_rows": self.total_rows,
            "total_cols": self.total_cols,
            "uploaded_at": self.uploaded_at.strftime("%b %d, %Y %I:%M %p"),
            "last_opened_at": self.last_opened_at.strftime("%b %d, %Y %I:%M %p"),
            "has_segmentation": bool(self.last_segmentation_result),
            "has_recommendation": bool(self.last_recommendation_result),
            "has_reviews": bool(self.last_review_result),
        }
