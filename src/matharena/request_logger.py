"""
Dirty global object for debug request logging.
"""

import json
import os
import tempfile
from loguru import logger
from collections import OrderedDict


class RequestLogger:
    def __init__(self):
        self.log_dir = os.environ.get("MATHARENA_REQUEST_LOG_DIR", "logs/requests")
        self.enabled = os.environ.get("MATHARENA_REQUEST_LOGGING", "1").lower() not in {"0", "false", "no", "off"}
        self.redact_images = os.environ.get("MATHARENA_REDACT_REQUEST_IMAGES", "0").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.comp_name = None
        self.solver_name = None
        self.batch_idx_to_problem_idx = None
        self.batch_idx_to_comp_name = None

    def set_metadata(self, comp_name, solver_name, batch_idx_to_problem_idx, batch_idx_to_comp_name=None):
        self.comp_name = comp_name
        self.solver_name = solver_name
        self.batch_idx_to_problem_idx = batch_idx_to_problem_idx
        self.batch_idx_to_comp_name = batch_idx_to_comp_name

    def _comp_name(self, batch_idx):
        if self.batch_idx_to_comp_name is None:
            return self.comp_name
        return self.batch_idx_to_comp_name.get(batch_idx, self.comp_name)

    def _redact_for_logging(self, value):
        if not self.redact_images:
            return value
        if isinstance(value, str) and value.startswith("data:image/") and ";base64," in value:
            prefix, encoded = value.split(";base64,", 1)
            return f"{prefix};base64,<redacted {len(encoded)} chars>"
        if isinstance(value, dict):
            return {key: self._redact_for_logging(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._redact_for_logging(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._redact_for_logging(item) for item in value)
        return value

    def _write_json(self, logfile, data):
        os.makedirs(os.path.dirname(logfile), exist_ok=True)
        tmpfile = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=os.path.dirname(logfile),
                prefix=f".{os.path.basename(logfile)}.",
                suffix=".tmp",
                delete=False,
            ) as f:
                tmpfile = f.name
                json.dump(data, f, indent=4)
            os.replace(tmpfile, logfile)
        finally:
            if tmpfile and os.path.exists(tmpfile):
                try:
                    os.unlink(tmpfile)
                except OSError:
                    pass

    def _logfile(self, ts, batch_idx):
        if self.comp_name is None:
            return -1, f"{self.log_dir}/uninitialized/{ts}_idx{batch_idx}.json"
        try:
            problem_idx = self.batch_idx_to_problem_idx[batch_idx]
        except Exception:
            problem_idx = 0
        return problem_idx, f"{self.log_dir}/{self.comp_name}/{self.solver_name}/{ts}_p{problem_idx}_idx{batch_idx}.json"

    def log_request(self, ts, batch_idx, request, **info):
        if not self.enabled:
            return
        try:
            problem_idx, logfile = self._logfile(ts, batch_idx)
            if os.path.exists(logfile):
                logger.warning(f"Can't log request, log file already exists: {logfile}")
                return

            data = OrderedDict(
                {
                    "comp_name": self._comp_name(batch_idx),
                    "solver_name": self.solver_name,
                    "timestamp": ts,
                    "problem_idx": problem_idx,
                    "batch_idx": batch_idx,
                    "request_info": info,
                    "request": self._redact_for_logging(request),
                }
            )
            self._write_json(logfile, data)
        except Exception as exc:
            logger.warning(f"Can't log request: {exc}")

    def log_response(self, ts, batch_idx, response=None, **info):
        if not self.enabled:
            return
        try:
            problem_idx, logfile = self._logfile(ts, batch_idx)
            if not os.path.exists(logfile):
                logger.warning(f"Can't log response, log file does not exist: {logfile}")
                return

            try:
                with open(logfile, "r", encoding="utf-8") as f:
                    data = json.load(f, object_pairs_hook=OrderedDict)
            except Exception as exc:
                logger.warning(f"Recovering response log after unreadable request log {logfile}: {exc}")
                data = OrderedDict(
                    {
                        "comp_name": self._comp_name(batch_idx),
                        "solver_name": self.solver_name,
                        "timestamp": ts,
                        "problem_idx": problem_idx,
                        "batch_idx": batch_idx,
                        "request_log_error": str(exc),
                    }
                )

            # Update the data with the response information
            data["response_info"] = info
            if response is not None:
                data["response"] = self._redact_for_logging(response)
            self._write_json(logfile, data)
        except Exception as exc:
            logger.warning(f"Can't log response: {exc}")


request_logger = RequestLogger()
