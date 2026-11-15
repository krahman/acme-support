import logging
import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.api.errors import error_response
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
            try:
                response = await call_next(request)
            except Exception:  # last resort: anything no handler claimed
                log.exception("unhandled error on %s %s", request.method, request.url.path)
                response = error_response(500, "internal_error", "Something went wrong on our side.")
            response.headers["x-request-id"] = rid
            ms = (time.perf_counter() - start) * 1000
            log.info("%s %s %s %.0fms", request.method, request.url.path, response.status_code, ms)
            return response
        finally:
            request_id_var.reset(token)
