import logging

from fastapi import Request
from fastapi.responses import JSONResponse

from app.errors import AppError
from app.logging import request_id_var

log = logging.getLogger("genai.errors")


def error_response(status_code: int, code: str, message: str) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "request_id": request_id_var.get()}}
    return JSONResponse(status_code=status_code, content=body)


async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    log.error("%s: %s", exc.code, exc.__cause__ or exc, exc_info=exc)
    return error_response(exc.status_code, exc.code, exc.message)
