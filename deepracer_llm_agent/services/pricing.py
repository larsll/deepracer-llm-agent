import json
import logging
from typing import Dict, Any, Optional, TypedDict

import boto3
from botocore.exceptions import ClientError

class TokenPricing(TypedDict):
    """Pricing rates for tokens"""
    prompt_rate: float      # Cost per 1000 prompt tokens
    completion_rate: float  # Cost per 1000 completion tokens

class PricingService:
    """
    Service to fetch and manage pricing information for LLM tokens
    """
    
    def __init__(self, region: Optional[str] = None):
        """
        Initialize the pricing service
        
        Args:
            region: AWS region (optional)
        """
        self.logger = logging.getLogger("PricingService")
        
        # Default pricing rates if API calls fail
        self.default_pricing: TokenPricing = {
            "prompt_rate": 0.002,  # Default fallback rate per 1000 tokens
            "completion_rate": 0.006  # Default fallback rate per 1000 tokens
        }
        
        # Hard-coded as pricing is only in us-east-1
        client_region = "us-east-1"
        
        # Initialize pricing client
        self.pricing_client = boto3.client('pricing', region_name=client_region)
        self.logger.debug(f"Initialized pricing client with region: {client_region}")
        
        # Initialize with default pricing
        self.current_pricing = self.default_pricing.copy()
    
    def get_pricing(self) -> TokenPricing:
        """
        Get the current token pricing
        
        Returns:
            The current token pricing information
        """
        return self.current_pricing.copy()
    
    def _get_model_name(self, model_id: str) -> str:
        """
        Get the model name from the model ID for pricing lookup
        
        Args:
            model_id: The model identifier
            
        Returns:
            The standardized model name
        """
        # Enhanced model mapping 
        model_mapping = {
            "amazon.nova-lite": "Nova Lite",
            "amazon.nova-pro": "Nova Pro",
            "anthropic.claude-3-sonnet": "Claude 3 Sonnet",
            "anthropic.claude-3-haiku": "Claude 3 Haiku",
            "anthropic.claude-3-opus": "Claude 3 Opus",
            "mistral.mistral-large": "Mistral Large",
            "mistral.pixtral-large": "Pixtral Large 25.02",
            "meta.llama3": "Llama 3",
        }
        
        # Extract model name from ARN if applicable
        model_name = "Unknown"
        if "arn:aws:bedrock" in model_id:
            parts = model_id.split("/")
            if parts and ":" in parts[-1]:
                model_name = parts[-1].split(":")[0]
                self.logger.debug(f"Extracted model name from ARN: {model_name}")
        else:
            model_name = model_id
        
        # Match model ID directly
        for key, value in model_mapping.items():
            if key in model_name:
                return value
        
        # Try partial matches for common model families
        for model_family in ["claude", "mistral", "nova", "llama"]:
            if model_family in model_name.lower():
                return model_family.capitalize()
        
        return "Unknown"  # Default fallback
    
    def reset_to_defaults(self) -> None:
        """Reset pricing to defaults"""
        self.current_pricing = self.default_pricing.copy()
        self.logger.debug("Pricing reset to defaults")
    
    async def load_model_pricing(
        self,
        model_id: str,
        region: str = "eu-central-1"
    ) -> TokenPricing:
        """
        Fetch pricing information from AWS Price List API for the specified model
        
        Args:
            model_id: The model identifier
            region: The AWS region to fetch pricing for (default: 'eu-central-1')
            
        Returns:
            Token pricing information
        """
        try:
            self.logger.debug(f"Fetching pricing data for model: {model_id} in region {region}")
            
            # Map model ID to service code
            service_code = "AmazonBedrock"
            
            # Get model family from model ID
            model = self._get_model_name(model_id)
            
            pricing_filters = [
                {'Type': 'TERM_MATCH', 'Field': 'model', 'Value': model},
                {'Type': 'TERM_MATCH', 'Field': 'regionCode', 'Value': region}
            ]
            
            self.logger.debug(f"Using model name for pricing lookup: {model}")
            
            try:
                response = self.pricing_client.get_products(
                    ServiceCode=service_code,
                    Filters=pricing_filters
                )
                
                if response.get('PriceList') and len(response['PriceList']) > 0:
                    found_prompt_price = False
                    found_completion_price = False
                    
                    self.logger.debug(f"Found {len(response['PriceList'])} pricing items to parse")
                    
                    # Use default pricing as starting point (in case we only find one of the rates)
                    new_pricing = self.default_pricing.copy()
                    
                    # Parse the pricing data
                    for price_item in response['PriceList']:
                        price_data = json.loads(price_item)
                        
                        # Check if this is input or output token pricing
                        usage_type = price_data.get('product', {}).get('attributes', {}).get('usagetype', '')
                        inference_type = price_data.get('product', {}).get('attributes', {}).get('inferenceType', '')
                        feature = price_data.get('product', {}).get('attributes', {}).get('feature', '')
                        
                        # Skip batch inference pricing if we're doing on-demand inference
                        if 'Batch' in feature and 'batch' not in model_id:
                            self.logger.debug(f"Skipping batch pricing: {usage_type}")
                            continue
                        
                        # Skip cache read pricing
                        if 'cache' in inference_type or 'cache' in usage_type:
                            self.logger.debug(f"Skipping cache pricing: {usage_type}")
                            continue
                        
                        # Extract pricing information from the price dimensions
                        on_demand = price_data.get('terms', {}).get('OnDemand', {})
                        if not on_demand:
                            continue
                            
                        on_demand_key = next(iter(on_demand), None)
                        if not on_demand_key:
                            continue
                            
                        price_dimensions = on_demand[on_demand_key].get('priceDimensions', {})
                        if not price_dimensions:
                            continue
                            
                        dimension_key = next(iter(price_dimensions), None)
                        if not dimension_key:
                            continue
                            
                        price_dimension = price_dimensions[dimension_key]
                        
                        if (not price_dimension or 
                            not price_dimension.get('pricePerUnit') or 
                            not price_dimension['pricePerUnit'].get('USD')):
                            self.logger.debug("Invalid price dimension structure")
                            continue
                        
                        price_per_unit = float(price_dimension['pricePerUnit']['USD'])
                        
                        # Determine if this is input or output token pricing
                        if (('Input' in inference_type or 'input' in usage_type) and 
                            'cache' not in inference_type and 'cache' not in usage_type):
                            new_pricing['prompt_rate'] = price_per_unit
                            self.logger.debug(f"Found input token price: ${price_per_unit}/1K tokens ({usage_type})")
                            found_prompt_price = True
                        elif (('Output' in inference_type or 'output' in usage_type) and 
                              'cache' not in inference_type and 'cache' not in usage_type):
                            new_pricing['completion_rate'] = price_per_unit
                            self.logger.debug(f"Found output token price: ${price_per_unit}/1K tokens ({usage_type})")
                            found_completion_price = True
                    
                    # Update the current pricing
                    self.current_pricing = new_pricing
                    
                    if found_prompt_price and found_completion_price:
                        self.logger.info(
                            f"Loaded pricing data: Input tokens ${self.current_pricing['prompt_rate']}/1K tokens, "
                            f"Output tokens ${self.current_pricing['completion_rate']}/1K tokens"
                        )
                    elif found_prompt_price:
                        self.logger.warning("Only found input token pricing. Using default for output tokens.")
                    elif found_completion_price:
                        self.logger.warning("Only found output token pricing. Using default for input tokens.")
                    else:
                        self.logger.warning(f"No applicable pricing data found for model {model_id}, using defaults")
                else:
                    self.logger.warning(f"No pricing data found for model {model_id}, using defaults")
            
            except ClientError as e:
                self.logger.warning(f"Error fetching pricing data: {e}. Using default pricing.")
            
            return self.current_pricing
            
        except Exception as e:
            self.logger.warning(f"Failed to load model pricing: {e}. Using default pricing.")
            return self.default_pricing
    
    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> Dict[str, float]:
        """
        Calculate the cost of token usage
        
        Args:
            prompt_tokens: Number of prompt tokens used
            completion_tokens: Number of completion tokens used
            
        Returns:
            Dict with prompt cost, completion cost, and total cost
        """
        prompt_cost = prompt_tokens * (self.current_pricing['prompt_rate'] / 1000)
        completion_cost = completion_tokens * (self.current_pricing['completion_rate'] / 1000)
        
        return {
            'prompt_cost': prompt_cost,
            'completion_cost': completion_cost,
            'total_cost': prompt_cost + completion_cost
        }

