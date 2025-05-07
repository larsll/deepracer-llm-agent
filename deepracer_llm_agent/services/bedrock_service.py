"""
This is a compatibility layer that re-exports the modular Bedrock implementation.
Existing code can continue using this file as the entry point.
"""

from .bedrock import BedrockService

# Export the BedrockService class as the default export
__all__ = ['BedrockService']
