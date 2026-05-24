"""Exceptions for resume and profile ingestion."""


class ResumeParsingError(Exception):
    """Base exception for resume parsing failures."""


class UnsupportedFileTypeError(ResumeParsingError):
    """Raised when the resume file type is not supported."""


class FileTooLargeError(ResumeParsingError):
    """Raised when the resume file exceeds the configured size limit."""


class ExtractionFailedError(ResumeParsingError):
    """Raised when LLM structured extraction fails after retries."""


class GitHubUserNotFoundError(ResumeParsingError):
    """Raised when a GitHub username does not exist."""
