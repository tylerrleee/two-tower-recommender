from fastapi import HTTPException, status

class ModelNotLoadedException(HTTPException):
    """ Raise when model is not loaded"""
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service unavailable."
        )

class InsufficientDataException(HTTPException):
    """ Raised when insufficient applicants provided"""
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )

class MatchingFailedException(HTTPException):
    """Raised when matching algorithm fails"""
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )

class InvalidCSVException(HTTPException):
    """Raised when CSV format is invalid"""
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail
        )