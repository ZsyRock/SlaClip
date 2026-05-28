"""SlaClip Opacus overlay bridge."""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from .privacy_engine import PrivacyEngine

__all__ = ["PrivacyEngine"]
