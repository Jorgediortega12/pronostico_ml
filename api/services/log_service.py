from typing import Optional
import logging

# Logging a consola (el microservicio ML no escribe en la tabla de auditoría;
# esa la administra el backend Node sobre su propia BD).
logger = logging.getLogger("pronostico_ml")
logging.basicConfig(level=logging.INFO)


def log_error(db=None, action: str = "", error: Optional[Exception] = None,
              user_id=None, details: Optional[str] = None, key_value=None):
    if details is None:
        details = f"Error: {str(error)}" if error else "No details provided"
    logger.error("[%s] %s", action, details)