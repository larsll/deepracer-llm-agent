import os
import json
import base64
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from .services import BedrockService, PricingService
from .utils.model_metadata import model_metadata, NeuralNetworkType
from .utils.json_extractor import extract_json_from_llm_response


class LLMAgent:
    """
    Agent that processes DeepRacer images using LLMs through AWS Bedrock.
    Takes camera images, converts them to base64, and sends them to an LLM
    along with a structured prompt and action space information.
    """

    def __init__(self, metadata_path: str):
        """
        Initialize the DeepRacer LLM Agent.

        Args:
            metadata_path: Path to the model metadata JSON file
        """
        self.logger = logging.getLogger("LLMAgent")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

        # Load and process metadata
        self.metadata_path = metadata_path
        self.metadata = model_metadata.load_model_metadata(metadata_path)

        # Initialize services
        aws_region = os.environ.get('AWS_REGION', 'eu-central-1')
        self.bedrock_service = BedrockService(region=aws_region)
        self.pricing_service = PricingService(region=aws_region)

        # Set model ID and validate
        self.model_id = self._get_model_id_from_metadata()
        if not self.model_id:
            raise ValueError(
                "No model ID specified in configuration or environment")

        # Set active model in the Bedrock service
        self.bedrock_service.set_model(self.model_id)

        # Set variables
        self.bedrock_service.active_handler.set_system_prompt(
            self.metadata["llm_config"].get("system_prompt", "You are an AI driver assistant."))
        self.bedrock_service.active_handler.set_max_context_messages(
            self.metadata["llm_config"].get("context_window", 0))
        self.bedrock_service.active_handler.set_action_space(
            self.metadata.get("action_space", None))
        self.bedrock_service.active_handler.set_action_space_type(
            self.metadata.get("action_space_type", None))

        # Load pricing data for the model
        self._load_pricing()

        # Track image processing count
        self.image_count = 0

        # Get context window size if specified
        self.max_context_messages = self.metadata.get(
            "llm_config", {}).get("context_window", 0)



        self.logger.info(
            f"🚗 DeepRacer LLM Agent initialized with model: {self.model_id} in {aws_region}")

    def _get_model_id_from_metadata(self) -> str:
        """Get the model ID from metadata with fallbacks to environment variables"""
        # First check if metadata has an LLM config with model_id
        if model_metadata.is_llm_model():
            llm_config = model_metadata.get_llm_config()
            if llm_config and "model_id" in llm_config:
                return llm_config["model_id"]

        # Then try environment variables as fallback
        return (os.environ.get('INFERENCE_PROFILE_ARN') or
                os.environ.get('DEFAULT_MODEL_ID') or
                'anthropic.claude-3-sonnet-20240229-v1:0')  # Default model

    def _load_pricing(self) -> None:
        """Load pricing data for the current model"""
        try:
            # For simplicity, we'll use the synchronous version instead of async
            # In a real implementation, you might want to handle this properly with async
            self.pricing_service.reset_to_defaults()  # Start with defaults
            # We'd ideally call the async method here, but we'll skip the actual API call for now
            # and just use defaults
        except Exception as e:
            self.logger.warning(f"Failed to initialize pricing: {e}")

    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Process an image and return an action using the LLM.

        Args:
            image_path: Path to the image file

        Returns:
            Dict containing steering angle and speed recommendations
        """
        try:
            # Track image count
            self.image_count += 1
            self.logger.debug(f"Processing image #{self.image_count}...")

            # Convert image to base64
            b64_image = self._image_to_base64(image_path)

            # Get the prompt from metadata
            llm_config = self.metadata.get("llm_config", {})
            prompt = llm_config.get("repeated_prompt",
                                    f"Analyze this image. This is image #{self.image_count}.")

            # Add context for subsequent images
            if self.image_count > 1 and self.max_context_messages > 0:
                prompt += " Compare with previous image to interpret how you are moving."

            # Process the image using our BedrockService
            try:
                # Process image with model
                action = self.bedrock_service.process(prompt, b64_image)

                # Log and validate the response
                self.logger.debug(f"Extracted driving action: {json.dumps(action)}")

                # Validate the action has required fields
                if 'steering_angle' not in action or 'speed' not in action:
                    self.logger.warning("Missing required driving parameters in response")

                    # Provide default values for missing parameters
                    if 'steering_angle' not in action:
                        action['steering_angle'] = 0.0  # Neutral steering
                    if 'speed' not in action:
                        action['speed'] = 1.0  # Safe default speed

                    # Add a flag to indicate this was a fallback action
                    action['fallback'] = True
                    action['error'] = "Missing required parameters in response"

                # Normalize action according to our action space
                return self._normalize_action(action)

            except Exception as e:
                self.logger.error(f"Failed to parse driving action: {e}")
                # Return safe defaults
                return {
                    "steering_angle": 0.0,
                    "speed": 1.0,
                    "fallback": True,
                    "error": f"Failed to process: {str(e)}"
                }

        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            # Return a safe default action
            return {"steering_angle": 0.0, "speed": 1.0}

    def _image_to_base64(self, image_path: str) -> str:
        """Convert an image to base64 encoded string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _parse_action(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract the driving action

        Args:
            response_text: The text response from the LLM

        Returns:
            Dict containing steering angle and speed values
        """
        try:
            # Try to extract JSON from the response text
            action = extract_json_from_llm_response(response_text, self.logger)

            # Validate the action has required fields
            if 'steering_angle' not in action or 'speed' not in action:
                self.logger.warning(
                    "Missing required driving parameters in response")

                # Provide default values for missing parameters
                if 'steering_angle' not in action:
                    action['steering_angle'] = 0.0  # Neutral steering
                if 'speed' not in action:
                    action['speed'] = 1.0  # Safe default speed

                # Add a flag to indicate this was a fallback action
                action['fallback'] = True
                action['error'] = "Missing required parameters in response"

            return action

        except Exception as e:
            self.logger.error(f"Failed to parse driving action: {e}")
            # Return safe defaults
            return {
                "steering_angle": 0.0,
                "speed": 1.0,
                "fallback": True,
                "error": f"Failed to parse: {str(e)}"
            }

    def _normalize_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize action values according to metadata action space

        Args:
            action: The raw action from LLM response

        Returns:
            Normalized action that fits within action space constraints
        """
        try:
            # Extract the values
            steering_angle = float(action.get('steering_angle', 0.0))
            speed = float(action.get('speed', 1.0))

            # Use our model metadata handler to normalize the action
            normalized = model_metadata.normalize_action(steering_angle, speed)

            # Preserve any metadata fields from the original action
            for key, value in action.items():
                if key not in normalized and key not in ('steering_angle', 'speed'):
                    normalized[key] = value

            return normalized

        except Exception as e:
            self.logger.error(f"Error normalizing action: {e}")
            return action  # Return original if normalization fails

    def get_token_usage(self) -> Dict[str, Any]:
        """Get the token usage statistics and cost information"""
        # Get token usage from the bedrock service
        token_usage = self.bedrock_service.get_token_usage()

        # Get current pricing
        pricing = self.pricing_service.get_pricing()

        # Calculate costs using the pricing service
        costs = self.pricing_service.calculate_cost(
            token_usage.get('input_tokens', 0),
            token_usage.get('output_tokens', 0)
        )

        # Return combined information
        return {
            "prompt_tokens": token_usage.get('input_tokens', 0),
            "completion_tokens": token_usage.get('output_tokens', 0),
            "total_tokens": token_usage.get('total_tokens', 0),
            "pricing": pricing,
            "estimated_cost": costs.get('total_cost', 0)
        }

    def reset(self, reset_tokens: bool = False, refresh_pricing: bool = False) -> None:
        """
        Reset the agent's conversation history and token counts

        Args:
            reset_tokens: Whether to reset token tracking
            refresh_pricing: Whether to refresh pricing data
        """
        # Clear conversation in bedrock service
        if hasattr(self.bedrock_service, 'clear_conversation'):
            self.bedrock_service.clear_conversation()

        self.image_count = 0

        if reset_tokens:
            self.logger.info(
                "🔄 DeepRacer agent reset (including token counts)")
            self.bedrock_service.reset_token_count()
        else:
            self.logger.info("🔄 DeepRacer agent reset")

        if refresh_pricing and model_metadata.is_llm_model():
            self._load_pricing()
