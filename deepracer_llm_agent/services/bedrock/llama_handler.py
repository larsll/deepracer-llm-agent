from typing import Dict, Any, Optional, List
import json

from .base_handler import ModelHandler
from deepracer_llm_agent.utils.model_metadata import ActionSpace, ActionSpaceType
from deepracer_llm_agent.utils.json_extractor import extract_json_from_llm_response

class LlamaHandler(ModelHandler):
    """Handler for Meta's Llama models on AWS Bedrock"""
    
    def __init__(self, model_id: str, region: Optional[str] = None):
        """
        Initialize the Llama handler
        
        Args:
            model_id: The Llama model ID (e.g., "meta.llama3-70b-instruct-v1:0")
            region: AWS region (optional)
        """
        super().__init__(model_id, region)
        
        # Llama-specific settings
        self.max_gen_len = 1024
        self.temperature = 0.0  # Deterministic for DeepRacer
        
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
        Prepare a prompt for Llama in the format it expects
        
        Args:
            text_prompt: The text prompt to send to Llama
            image_data: Optional base64-encoded image data
            
        Returns:
            Dict containing the formatted prompt for Llama
        """
        # Llama uses a different formatting style for system prompts and conversation
        full_prompt = f"<|system|>\n{self.system_prompt}\n"
        
        # Add action space information if available
        if self.action_space is not None and self.action_space_type is not None:
            full_prompt += f"\nAction space type: {self.action_space_type}\n"
            full_prompt += f"Action space: {json.dumps(self.action_space)}\n"
        
        full_prompt += "</s>\n"
        
        # Add conversation context if available
        if self.conversation_context and self.max_context_messages > 0:
            # Format the conversation history in Llama's expected format
            history = self.conversation_context[-self.max_context_messages*2:]  # *2 for user/assistant pairs
            for entry in history:
                role = entry.get('role', '')
                content = entry.get('content', '')
                
                if role == 'user':
                    full_prompt += f"<|user|>\n{content}\n</s>\n"
                elif role == 'assistant':
                    full_prompt += f"<|assistant|>\n{content}\n</s>\n"
        
        # Add the current prompt
        full_prompt += f"<|user|>\n{text_prompt}\n</s>\n"
        full_prompt += "<|assistant|>\n"
        
        # Llama request format 
        request_body = {
            "prompt": full_prompt,
            "max_gen_len": self.max_gen_len,
            "temperature": self.temperature,
        }
        
        # Add image if provided
        if image_data:
            request_body["image_data"] = [image_data]
        
        return request_body
    
    def extract_response_text(self, response_body: Dict[str, Any]) -> str:
        """
        Extract the text response from Llama's response body
        
        Args:
            response_body: The parsed JSON response from Llama
            
        Returns:
            The extracted text response
        """
        # Llama typically returns a generation field
        response_text = response_body.get("generation", "")
        
        # Store conversation history if tracking context
        if self.max_context_messages > 0:
            # Add the assistant message to history
            self.conversation_context.append({
                "role": "assistant",
                "content": response_text
            })
            
            # Limit conversation context if needed
            if len(self.conversation_context) > self.max_context_messages * 2:  # *2 for user/assistant pairs
                self.conversation_context = self.conversation_context[-self.max_context_messages*2:]
        
        return response_text
    
    def update_token_count(self, response_body: Dict[str, Any]) -> None:
        """
        Update token counts based on Llama's response
        
        Args:
            response_body: The parsed JSON response from Llama
        """
        # Llama may have a different format for token usage
        usage = response_body.get("usage", {})
        
        # Various formats seen in different Llama versions
        self.input_tokens += usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0) or usage.get("generated_tokens", 0)
    
    def extract_driving_action(self, response_text: str) -> Dict[str, Any]:
        """
        Extract the driving action from Llama's text response
        
        Args:
            response_text: The text response from Llama
            
        Returns:
            Dict containing the driving action
        """
        return extract_json_from_llm_response(response_text, self.logger, "Llama")
    
    def process(self, prompt: str, image_data: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a prompt with image and return a driving action
        
        Args:
            prompt: The text prompt
            image_data: Base64-encoded image data
            
        Returns:
            Dict containing the driving action
        """
        # Track the user message in conversation history if needed
        if self.max_context_messages > 0:
            self.conversation_context.append({
                "role": "user",
                "content": prompt
            })
            
        # Prepare the prompt
        request_body = self.prepare_prompt(prompt, image_data)
        
        # Invoke the model
        response_body = self.invoke_model(request_body)
        
        # Extract the text response
        response_text = self.extract_response_text(response_body)
        
        # Extract and return the driving action
        return self.extract_driving_action(response_text)
