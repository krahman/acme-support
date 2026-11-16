class AppError(Exception):
    """An expected failure with a client-safe message. Never put provider details in `message`."""

    status_code = 500
    code = "internal_error"
    message = "Something went wrong on our side."


class UpstreamTimeoutError(AppError):
    status_code = 504
    code = "upstream_timeout"
    message = "The AI provider took too long to respond. Please try again."


class UpstreamUnavailableError(AppError):
    status_code = 503
    code = "upstream_unavailable"
    message = "The AI provider is temporarily unavailable. Please try again shortly."


class UpstreamRejectedError(AppError):
    status_code = 502
    code = "upstream_rejected"
    message = "The AI provider rejected the request."


class AgentLoopError(AppError):
    status_code = 500
    code = "agent_loop"
    message = "The assistant could not finish answering. Please rephrase and try again."


class StorageUnavailableError(AppError):
    status_code = 503
    code = "storage_unavailable"
    message = "Conversation storage is temporarily unavailable. Please try again shortly."


class SessionBusyError(AppError):
    status_code = 409
    code = "session_busy"
    message = "Your previous message in this conversation is still being answered. Please wait and try again."


class KnowledgeBaseUnavailableError(AppError):
    status_code = 503
    code = "knowledge_base_unavailable"
    message = "The knowledge base is temporarily unavailable. Please try again shortly."
