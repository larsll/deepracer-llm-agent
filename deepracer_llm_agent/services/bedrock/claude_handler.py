from typing import Dict, Any, Optional, List
import json

from .base_handler import ModelHandler
from deepracer_llm_agent.utils.model_metadata import ActionSpace, ActionSpaceType
from deepracer_llm_agent.utils.json_extractor import extract_json_from_llm_response

class ClaudeHandler(ModelHandler):
    """Handler for Anthropic Claude models on AWS Bedrock"""
    
    def __init__(self, model_id: str, region: Optional[str] = None):
        """
        Initialize the Claude handler
        
        Args:
            model_id: The Claude model ID (e.g., "anthropic.claude-3-sonnet-20240229-v1:0")
            region: AWS region (optional)
        """
        super().__init__(model_id, region)
        
        # Claude-specific settings
        self.max_tokens = 1024
        self.temperature = 0.0  # Deterministic for DeepRacer
        self.anthropic_version = "bedrock-2023-05-31"
        
        # Add conversation context tracking
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
    
    def prepare_prompt(self, text_prompt: str, image_data: Optional[str] = None) -> Dict[str, Any]:
        """
        Prepare a prompt for Claude in the format it expects
        
        Args:
            text_prompt: The text prompt to send to Claude
            image_data: Optional base64-encoded image data
            
        Returns:
            Dict containing the formatted prompt for Claude
        """
        messages = []
        
        # Add system message with action space if available
        system_content = []
        system_content.append({
            "type": "text",
            "text": self.system_prompt
        })
        
        if self.action_space is not None and self.action_space_type is not None:
            action_space_text = (f"Action space type: {self.action_space_type}\n"
                                f"Action space: {json.dumps(self.action_space)}")
            system_content.append({
                "type": "text",
                "text": action_space_text
            })
            
        # Add the system message
        messages.append({
            "role": "system",
            "content": system_content
        })
        
        # Add conversation context if available
        if self.conversation_context and self.max_context_messages > 0:
            context_messages = self.conversation_context[-self.max_context_messages*2:]  # *2 for pairs
            messages.extend(context_messages)
            
        # Create the user message content
        user_content = []
        
        # Add text prompt
        user_content.append({
            "type": "text",
            "text": text_prompt
        })
        
        # Add image if provided
        if image_data:
            user_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_data
                }
            })
        
        # Add the user message
        user_message = {
            "role": "user",
            "content": user_content
        }
        
        messages.append(user_message)
        
        # Claude message format
        return {
            "anthropic_version": self.anthropic_version,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": messages
        }
    
    def extract_response_text(self, response_body: Dict[str, Any]) -> str:
        """
        Extract the text response from Claude's response body
        
        Args:
            response_body: The parsed JSON response from Claude
            
        Returns:
            The extracted text response
        """
        # Claude response format has content as a list of blocks
        content = response_body.get("content", [])
        
        # Extract text from all text blocks
        response_text = ""
        for block in content:
            if block.get("type") == "text":
                response_text += block.get("text", "")
        
        # Store conversation history if tracking context
        if self.max_context_messages > 0:
            # Add the assistant message to history
            assistant_message = {
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": response_text
                }]
            }
            
            self.conversation_context.append(assistant_message)
            
            # Limit conversation context if needed
            if len(self.conversation_context) > self.max_context_messages * 2:  # *2 for user/assistant pairs
                self.conversation_context = self.conversation_context[-self.max_context_messages*2:]
        
        return response_text
    
    def update_token_count(self, response_body: Dict[str, Any]) -> None:
        """
        Update token counts based on Claude's response
        
        Args:
            response_body: The parsed JSON response from Claude
        """
        usage = response_body.get("usage", {})
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
    
    def extract_driving_action(self, response_text: str) -> Dict[str, Any]:
        """
        Extract the driving action from Claude's text response
        
        Args:
            response_text: The text response from Claude
            
        Returns:
            Dict containing the driving action
        """
        return extract_json_from_llm_response(response_text, self.logger, "Claude")
    
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
        user_content = [{
            "type": "text",
            "text": prompt
        }]
        
        # Add user message to conversation history if needed
        if self.max_context_messages > 0:
            user_message = {
                "role": "user",
                "content": user_content
            }
            self.conversation_context.append(user_message)
            
        # Prepare the prompt
        request_body = self.prepare_prompt(prompt, image_data)
        
        # Invoke the model
        response_body = self.invoke_model(request_body)
        
        # Extract the text response
        response_text = self.extract_response_text(response_body)
        
        # Extract and return the driving action
        return self.extract_driving_action(response_text)

