"""Small standard-library HTTP adapter for the existing EduPredict engine."""

from __future__ import annotations

import io
import json
import logging
import math
import mimetypes
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import pandas as pd

from data_loader import EARLY_FEATURES, NUMERIC_RANGES, VALID_CATEGORICAL_VALUES
from predict import EduPredictor, PredictionValidationError

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
try:
    predictor = EduPredictor(models_dir=ROOT / "models")
    predictor_error = None
except Exception as exc:
    logging.exception("EduPredict prediction artifacts failed to load")
    predictor = None
    predictor_error = str(exc)
DATASET = ROOT / "data" / "student-mat.csv"


def dataset_frame() -> pd.DataFrame:
    return pd.read_csv(DATASET, sep=";")


def validate_frame(frame: pd.DataFrame) -> list[dict]:
    problems: list[dict] = []
    missing = [name for name in EARLY_FEATURES if name not in frame.columns]
    problems.extend({"row": None, "column": name, "message": "Required model input column is missing."} for name in missing)
    for row_idx, row in frame.iterrows():
        for name in EARLY_FEATURES:
            if name not in frame.columns:
                continue
            value = row[name]
            row_number = int(row_idx) + 2
            if pd.isna(value) or str(value).strip() == "":
                problems.append({"row": row_number, "column": name, "message": "Value is required."})
                continue
            if name in VALID_CATEGORICAL_VALUES:
                allowed = VALID_CATEGORICAL_VALUES[name]
                if str(value).strip() not in allowed:
                    problems.append({"row": row_number, "column": name, "value": str(value), "message": f"Use one of: {', '.join(allowed)}."})
                continue
            try:
                number = float(value)
                lower, upper = NUMERIC_RANGES[name]
                if not math.isfinite(number) or number < lower or number > upper:
                    raise ValueError
                if not number.is_integer():
                    problems.append({"row": row_number, "column": name, "value": str(value), "message": "Enter a whole number."})
            except (ValueError, TypeError):
                problems.append({"row": row_number, "column": name, "value": str(value), "message": f"Enter a whole number from {NUMERIC_RANGES[name][0]} to {NUMERIC_RANGES[name][1]}."})
        if "G3" in frame.columns and not pd.isna(row["G3"]) and str(row["G3"]).strip():
            try:
                observed = float(row["G3"])
                if not math.isfinite(observed) or observed < 0 or observed > 20 or not observed.is_integer():
                    raise ValueError
            except (ValueError, TypeError):
                problems.append({"row": row_number, "column": "G3", "value": str(row["G3"]), "message": "Observed G3 must be a whole number from 0 to 20. It is retained for context only and is never used to predict."})
        if len(problems) >= 200:
            problems.append({"row": None, "column": None, "message": "Showing the first 200 validation issues."})
            break
    return problems


def build_model_input(frame: pd.DataFrame) -> pd.DataFrame:
    """Select and type only the model's 30 early-prediction features."""
    model_input = frame[EARLY_FEATURES].copy()
    for name in NUMERIC_RANGES:
        if name in EARLY_FEATURES:
            model_input[name] = pd.to_numeric(model_input[name], errors="raise").astype(int)
    return model_input


def build_batch_result(frame: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """Preserve identifiers while keeping observed targets distinct and out of model inputs."""
    result = frame.drop(columns=["G1", "G2", "G3", "academic_risk"], errors="ignore").copy()
    if "G3" in frame.columns:
        result["observed_G3"] = pd.to_numeric(frame["G3"], errors="coerce")
    if "academic_risk" in frame.columns:
        result["observed_academic_risk"] = frame["academic_risk"]
    for name in ("predicted_grade", "academic_risk_class", "risk_probability", "risk_status", "risk_tier"):
        result[name] = predictions[name].values
    return result


class EduPredictHandler(BaseHTTPRequestHandler):
    server_version = "EduPredict/1.0"

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/api/health":
            self.send_json(200, {"status": "ok" if predictor else "degraded", "scenario": "early_prediction", "prediction_available": predictor is not None, "features": EARLY_FEATURES})
            return
        if path == "/api/schema":
            self.send_json(200, {"features": EARLY_FEATURES, "categorical": VALID_CATEGORICAL_VALUES, "numeric_ranges": NUMERIC_RANGES, "excluded": ["G1", "G2", "G3", "academic_risk"]})
            return
        if path == "/api/dashboard":
            try:
                frame = dataset_frame()
                grades = frame["G3"].astype(int)
                with open(ROOT / "outputs" / "metrics" / "regression_metrics.json", encoding="utf-8") as handle:
                    regression_metrics = json.load(handle)
                with open(ROOT / "outputs" / "metrics" / "classification_metrics.json", encoding="utf-8") as handle:
                    classification_metrics = json.load(handle)
                with open(ROOT / "outputs" / "metrics" / "eda_summary.json", encoding="utf-8") as handle:
                    eda_summary = json.load(handle)
                self.send_json(200, {
                    "records": int(len(frame)),
                    "average_observed_grade": round(float(grades.mean()), 3),
                    "pass_count": int((grades >= 10).sum()),
                    "at_risk_count": int((grades < 10).sum()),
                    "grade_counts": {str(i): int((grades == i).sum()) for i in range(21)},
                    "dataset_name": "UCI Student Performance (Mathematics)",
                    "selected_models": {"regression": "Random Forest Regressor", "classification": "Logistic Regression"},
                    "test_metrics": {"regression": regression_metrics["test_evaluation"], "classification": classification_metrics["test_evaluation"]},
                    "cross_validation": {"regression": regression_metrics["cv_5fold"], "classification": classification_metrics["cv_5fold"]},
                    "correlations": eda_summary.get("top_correlations_with_G3", {}),
                })
            except Exception:
                logging.exception("Dashboard data failed to load")
                self.send_json(500, {"error": "Dashboard data could not be loaded."})
            return
        if path in ("/api/sample-dataset", "/api/template", "/api/template/example"):
            try:
                if path == "/api/sample-dataset":
                    payload = DATASET.read_bytes()
                    filename = "student-mat.csv"
                else:
                    frame = dataset_frame()
                    columns = EARLY_FEATURES
                    template = frame[columns].head(1) if path.endswith("example") else pd.DataFrame(columns=columns)
                    payload = template.to_csv(index=False).encode("utf-8")
                    filename = "edupredict-template-example.csv" if path.endswith("example") else "edupredict-template-headers.csv"
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            except Exception:
                self.send_json(500, {"error": "Requested CSV could not be prepared."})
            return
        relative = "index.html" if path == "/" else path.lstrip("/")
        target = (WEB / relative).resolve()
        if not target.is_relative_to(WEB.resolve()) or not target.is_file():
            self.send_error(404, "Not found")
            return
        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 64 * 1024 * 1024:
            self.send_json(400, {"error": "Request body is empty or exceeds the 64 MB limit."})
            return
        body = self.rfile.read(length)
        if path == "/api/predict":
            self.handle_single(body)
        elif path == "/api/predict/batch":
            self.handle_batch(body)
        elif path == "/api/validate/batch":
            self.handle_batch(body, validate_only=True)
        else:
            self.send_json(404, {"error": "Endpoint not found."})

    def handle_single(self, body: bytes) -> None:
        try:
            record = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json(400, {"error": "Request body must be valid JSON."})
            return
        if not isinstance(record, dict):
            self.send_json(400, {"error": "Request body must be a JSON object."})
            return
        if predictor is None:
            self.send_json(503, {"error": "Saved prediction models are unavailable. Check the model artifacts and required Python packages."})
            return
        try:
            result = predictor.predict_single(record, include_explanation=True)
            self.send_json(200, result)
        except PredictionValidationError as exc:
            self.send_json(400, {"error": str(exc)})
        except Exception:
            logging.exception("Single prediction failed")
            self.send_json(500, {"error": "Prediction could not be completed. Check that the model artifacts are available."})

    def read_csv_upload(self, body: bytes) -> pd.DataFrame | None:
        try:
            content_type = self.headers.get("Content-Type", "").encode("ascii", errors="strict")
            message = BytesParser(policy=policy.default).parsebytes(
                b"Content-Type: " + content_type + b"\r\nMIME-Version: 1.0\r\n\r\n" + body
            )
            upload_bytes = None
            for part in message.iter_parts():
                if part.get_content_disposition() == "form-data" and part.get_param("name", header="content-disposition") == "file":
                    upload_bytes = part.get_payload(decode=True)
                    break
            if not upload_bytes:
                return None
            return pd.read_csv(io.BytesIO(upload_bytes), sep=None, engine="python", dtype=object)
        except Exception:
            return None

    def handle_batch(self, body: bytes, validate_only: bool = False) -> None:
        frame = self.read_csv_upload(body)
        if frame is None:
            self.send_json(400, {"error": "Choose a readable CSV file to upload."})
            return
        if frame.empty:
            self.send_json(400, {"error": "The CSV contains headers but no student rows."})
            return
        if len(frame) > 5000:
            self.send_json(400, {"error": "Upload no more than 5,000 rows at a time."})
            return
        problems = validate_frame(frame)
        if validate_only:
            self.send_json(200, {"valid": not problems, "count": len(frame), "columns": list(frame.columns), "preview": json.loads(frame.head(8).to_json(orient="records")), "problems": problems})
            return
        if predictor is None:
            self.send_json(503, {"error": "Saved prediction models are unavailable. Check the model artifacts and required Python packages."})
            return
        if problems:
            self.send_json(400, {"error": "Fix the listed CSV issues before running predictions.", "problems": problems})
            return
        try:
            # Only the exact early-prediction feature set reaches the trained model.
            model_input = build_model_input(frame)
            predictions = predictor.predict_batch(model_input)
            result = build_batch_result(frame, predictions)
            rows = json.loads(result.to_json(orient="records"))
            observed_available = "G3" in frame.columns and bool(frame["G3"].apply(lambda value: not pd.isna(value) and str(value).strip() != "").any())
            self.send_json(200, {"columns": list(result.columns), "rows": rows, "count": len(result), "observed_grade_available": observed_available})
        except Exception:
            logging.exception("Batch prediction failed")
            self.send_json(400, {"error": "Batch prediction failed. Check column values and accepted categories."})


if __name__ == "__main__":
    print("EduPredict UI and API listening at http://127.0.0.1:5000")
    ThreadingHTTPServer(("127.0.0.1", 5000), EduPredictHandler).serve_forever()
