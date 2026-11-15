import logging
import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.logging import request_id_var

log = logging.getLogger("genai.access")

_VALID_RID = re.compile(r"^[A-Za-z0-9-]{1,64}$")


class AccessLogMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        incoming = request.headers.get("x-request-id", "")
        rid = incoming if _VALID_RID.match(incoming) else str(uuid.uuid4())
        token = request_id_var.set(rid)
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            ms = (time.perf_counter() - start) * 1000
            log.exception("%s %s 500 %.0fms", request.method, request.url.path, ms)
            raise
        else:
            ms = (time.perf_counter() - start) * 1000
            response.headers["x-request-id"] = rid
            log.info("%s %s %s %.0fms", request.method, request.url.path, response.status_code, ms)
            return response
        finally:
            request_id_var.reset(token)
