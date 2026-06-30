"""
Custom exceptions for the Person Locator project.
This module defines a hierarchy of errors for Network, GUI, and Mathematics.
"""

from typing import Optional

class PersonLocatorError(Exception):
    """Base exception for the Person Locator project."""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class NetworkError(PersonLocatorError):
    """Exceptions related to networking operations."""
    def __init__(self, message: str, endpoint: Optional[str] = None, details: Optional[str] = None):
        msg = f"[NETZWERK] {message}"
        if endpoint: msg += f" | Endpunkt: {endpoint}"
        if details: msg += f" | Details: {details}"
        super().__init__(msg)
        self.endpoint = endpoint
        self.details = details

class ConnectionError(NetworkError):
    """Raised when a connection cannot be established or is lost."""
    def __init__(self, endpoint: str, details: Optional[str] = None):
        super().__init__("Verbindungsfehler", endpoint=endpoint, details=details)

class ProtocolError(NetworkError):
    """Raised when there is a mismatch or error in the communication protocol."""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message, details=details)


class MathError(PersonLocatorError):
    """Exceptions related to mathematical calculations and triangulation."""
    def __init__(self, message: str, operation: Optional[str] = None, details: Optional[str] = None):
        msg = f"[MATHEMATIK] {message}"
        if operation: msg += f" | Operation: {operation}"
        if details: msg += f" | Details: {details}"
        super().__init__(msg)
        self.operation = operation
        self.details = details

class CalibrationError(MathError):
    """Raised when camera calibration or PnP calculation fails."""
    def __init__(self, camera_name: str, reason: str):
        super().__init__(
            f"Kalibrierung für Kamera '{camera_name}' fehlgeschlagen", 
            operation="PnP / Camera Calibration", 
            details=reason
        )

class TriangulationError(MathError):
    """Raised when 3D triangulation fails or yields invalid results."""
    def __init__(self, person_id: str, reason: str):
        super().__init__(
            f"Triangulierung für Person '{person_id}' fehlgeschlagen",
            operation="Triangulation",
            details=reason
        )


class GUIError(PersonLocatorError):
    """Exceptions related to the GUI or rendering components."""
    def __init__(self, message: str, component: Optional[str] = None, details: Optional[str] = None):
        msg = f"[GUI] {message}"
        if component: msg += f" | Komponente: {component}"
        if details: msg += f" | Details: {details}"
        super().__init__(msg)
        self.component = component
        self.details = details

class RenderingError(GUIError):
    """Raised when a rendering operation fails."""
    def __init__(self, component: str, reason: str):
        super().__init__("Rendering-Fehler", component=component, details=reason)

class ConfigUIError(GUIError):
    """Raised when there is an error in the GUI configuration or state."""
    def __init__(self, message: str, component: Optional[str] = None):
        super().__init__(message, component=component)
