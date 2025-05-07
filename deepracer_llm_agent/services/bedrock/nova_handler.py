import os
import json
import logging
from typing import Dict, Any, Optional, List, Union

from .base_handler import ModelHandler
from deepracer_llm_agent.utils.model_metadata import ActionSpace, ActionSpaceType
from deepracer_llm_agent.utils.json_extractor import extract_json_from_llm_response


class NovaHandler(ModelHandler):
    """Handler for Amazon Nova models on AWS Bedrock"""

    def __init__(self, model_id: str, region: Optional[str] = None):
        """
        Initialize the Nova handler

        Args:
            model_id: The Nova model ID (e.g., "amazon.nova-pro-v1")
            region: AWS region (optional)
        """
        super().__init__(model_id, region)

        # Nova-specific settings
        self.system_prompt = "You are an AI driver assistant."
        self.max_context_messages = 0
        self.conversation_context = []
        self.action_space = None
        self.action_space_type = None

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt for the model"""
        self.system_prompt = prompt

    def set_max_context_messages(self, max_messages: int) -> None:
        """Set maximum number of conversation context messages to maintain"""
        self.max_context_messages = max_messages

    def set_action_space(self, action_space: ActionSpace) -> None:
        """Set the action space for the model"""
        self.action_space = action_space

    def set_action_space_type(self, action_space_type: ActionSpaceType) -> None:
        """Set the action space type for the model"""
        self.action_space_type = action_space_type

    def clear_conversation(self) -> None:
        """Clear the conversation context"""
        self.conversation_context = []

    def _create_user_message(self, prompt: str, image_data: Optional[str]) -> Dict[str, Any]:
        """
        Create a user message with optional image for Nova format

        Args:
            prompt: The text prompt
            image_data: Optional base64-encoded image

        Returns:
            Message in Nova format
        """
        content = [{"text": prompt}]

        if image_data:
            content.append({
                "image": {
                    "format": "jpeg",
                    "source": {
                        "bytes": image_data
                    }
                }
            })

        return {
            "role": "user",
            "content": content
        }

    def prepare_prompt(self, text_prompt: str, image_data: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare a prompt for Nova in the format it expects

        Args:
            text_prompt: The text prompt to send to Nova
            image_data: Optional base64-encoded image data

        Returns:
            Dict containing the formatted prompt for Nova
        """
        # Create the user message with image
        user_message = self._create_user_message(text_prompt, image_data)

        # Initial system message with action space
        system_message = {
            "role": "user",
            "content": [
                {"text": self.system_prompt},
                {"text": json.dumps({
                    "action_space_type": self.action_space_type,
                    "action_space": self.action_space
                })}
            ]
        }

        # Build the messages array with context if available
        messages = [system_message]

        if self.conversation_context and self.max_context_messages > 0:
            # Add conversation context, limiting to max messages
            messages.extend(
                self.conversation_context[-self.max_context_messages:])

        # Add the current user message
        messages.append(user_message)

        # Nova payload structure
        return {
            "inferenceConfig": {
                "max_new_tokens": int(os.environ.get("MAX_TOKENS", "1000"))
            },
            "messages": messages
        }

    def extract_response_text(self, response_body: Dict[str, Any]) -> str:
        """
        Extract the text response from Nova's response body

        Args:
            response_body: The parsed JSON response from Nova

        Returns:
            The extracted text response
        """
        # Save response to conversation history if using context
        if self.max_context_messages > 0:
            # Get the text content
            content_text = ""

            if (response_body.get("output") and
                response_body["output"].get("message") and
                    response_body["output"]["message"].get("content")):
                content_text = response_body["output"]["message"]["content"][0].get(
                    "text", "")
            else:
                content_text = json.dumps(response_body)

            # Create assistant message
            assistant_message = {
                "role": "assistant",
                "content": [{"text": content_text}]
            }

            # Add to conversation context
            self.conversation_context.append(assistant_message)

            # Limit conversation context if needed
            if self.max_context_messages > 0:
                self.conversation_context = self.conversation_context[-self.max_context_messages:]

        # Extract and return the actual text
        if (response_body.get("output") and
            response_body["output"].get("message") and
                response_body["output"]["message"].get("content")):
            return response_body["output"]["message"]["content"][0].get("text", "")
        else:
            self.logger.error(
                f"Unexpected Nova response structure: {json.dumps(response_body)[:200]}")
            raise ValueError("Unexpected Nova response structure")

    def update_token_count(self, response_body: Dict[str, Any]) -> None:
        """
        Update token counts based on Nova's response

        Args:
            response_body: The parsed JSON response from Nova
        """
        if response_body.get("usage"):
            usage = response_body["usage"]
            self.input_tokens += usage.get("inputTokens", 0)
            self.output_tokens += usage.get("outputTokens", 0)
        else:
            self.logger.debug(
                "Could not determine token usage from Nova response")

    def extract_driving_action(self, response_text: str) -> Dict[str, Any]:
        """
        Extract the driving action from Nova's text response

        Args:
            response_text: The text response from Nova

        Returns:
            Dict containing the driving action
        """
        return extract_json_from_llm_response(response_text, self.logger, "Nova")

    def process(self, prompt: str, image_data: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a prompt with image and return a driving action

        Args:
            prompt: The text prompt
            image_data: Base64-encoded image data

        Returns:
            Dict containing the driving action
        """
        # Create the user message and add to conversation context if tracking
        user_message = self._create_user_message(prompt, image_data)
        if self.max_context_messages > 0:
            self.conversation_context.append(user_message)

        # Prepare the prompt
        request_body = self.prepare_prompt(prompt, image_data)

        # Invoke the model
        response_body = self.invoke_model(request_body)

        # Extract the text response
        response_text = self.extract_response_text(response_body)

        # Extract and return the driving action
        return self.extract_driving_action(response_text)
