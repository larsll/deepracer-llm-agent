import logging
from typing import Dict, Any, Optional, Union

from .base_handler import ModelHandler
from .claude_handler import ClaudeHandler
from .mistral_handler import MistralHandler
from .llama_handler import LlamaHandler
from .nova_handler import NovaHandler


class BedrockService:
    """
    Main service for interacting with AWS Bedrock models.
    Provides model selection and common interface for all supported models.
    """

    def __init__(self, region: Optional[str] = None):
        """
        Initialize the Bedrock service

        Args:
            region: AWS region (optional - uses boto3 default otherwise)
        """
        self.logger = logging.getLogger("BedrockService")
        self.region = region
        self.active_handler: Optional[ModelHandler] = None

    def get_handler_for_model(self, model_id: str) -> ModelHandler:
        """
        Get the appropriate handler for the given model ID

        Args:
            model_id: The Bedrock model ID

        Returns:
            An appropriate model handler instance

        Raises:
            ValueError: If the model type is not supported
        """
        model_prefix = model_id.split('/')[1].lower()

        if "anthropic" in model_prefix or "claude" in model_id.lower():
            return ClaudeHandler(model_id, self.region)
        elif "mistral" in model_prefix:
            return MistralHandler(model_id, self.region)
        elif "meta" in model_prefix or "llama" in model_id.lower():
            return LlamaHandler(model_id, self.region)
        elif "amazon" in model_prefix or "nova" in model_id.lower():
            return NovaHandler(model_id, self.region)
        else:
            raise ValueError(f"Unsupported model type: {model_id}")

    def set_model(self, model_id: str) -> None:
        """
        Set the active model by ID

        Args:
            model_id: The Bedrock model ID to use
        """
        self.active_handler = self.get_handler_for_model(model_id)
        self.logger.info(f"Set active model to {model_id}")

    def process(self, prompt: str, image_data: Optional[str] = None, model_id: Optional[str] = None) -> str:
        """
        Process a prompt with an LLM model

        Args:
            prompt: The text prompt to process
            image_data: Optional base64-encoded image data
            model_id: Optional model ID to use for this request only

        Returns:
            The model's text response

        Raises:
            ValueError: If no model is set and none is provided
        """
        # Use provided model or fall back to active handler
        handler = None
        if model_id:
            handler = self.get_handler_for_model(model_id)
        elif self.active_handler:
            handler = self.active_handler
        else:
            raise ValueError("No model specified and no active model set")

        return handler.process(prompt, image_data)

    def get_token_usage(self) -> Dict[str, int]:
        """
        Get token usage from the active model handler

        Returns:
            Dict with token usage statistics
        """
        if not self.active_handler:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        return self.active_handler.get_token_usage()

    def reset_token_count(self) -> None:
        """Reset token counters in the active handler"""
        if self.active_handler:
            self.active_handler.reset_token_count()
