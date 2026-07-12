import logging
import json
import sys


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        # Add extra request/prediction properties if present
        for key in ("path", "method", "status_code", "duration_ms", "prediction", "confidence"):
            if hasattr(record, key):
                log_record[key] = getattr(record, key)
        return json.dumps(log_record)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    # Ensure it doesn't propagate logs up to default handlers multiple times
    logger.propagate = False
    return logger
