import json
import os
import logging
import math
from enum import Enum
from typing import Dict, List, Union, Optional, TypedDict, Any, Tuple

# Configure logger
logger = logging.getLogger("ModelMetadata")

# Enums that mirror the TypeScript implementation
class ActionSpaceType(str, Enum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"

class NeuralNetworkType(str, Enum):
    DEEP_CONVOLUTIONAL_NETWORK_SHALLOW = "DEEP_CONVOLUTIONAL_NETWORK_SHALLOW"
    DEEP_CONVOLUTIONAL_NETWORK = "DEEP_CONVOLUTIONAL_NETWORK"
    LLM = "LLM"

class TrainingAlgorithm(str, Enum):
    CLIPPED_PPO = "clipped_ppo"
    SAC = "sac"

class SensorType(str, Enum):
    FRONT_FACING_CAMERA = "FRONT_FACING_CAMERA"
    STEREO_CAMERAS = "STEREO_CAMERAS"
    LIDAR = "LIDAR"

# Type definitions for action spaces and metadata
class DiscreteAction(TypedDict):
    steering_angle: float
    speed: float

class ContinuousRange(TypedDict):
    low: float
    high: float

class ContinuousActionSpace(TypedDict):
    speed: ContinuousRange
    steering_angle: ContinuousRange

# Define ActionSpace as a Union type
ActionSpace = Union[List[DiscreteAction], ContinuousActionSpace]

class LLMConfig(TypedDict):
    model_id: str
    max_tokens: int
    system_prompt: Union[str, List[str]]
    repeated_prompt: str
    context_window: int

# Model metadata type
class ModelMetadata(TypedDict, total=False):
    action_space: ActionSpace
    sensor: List[SensorType]
    neural_network: NeuralNetworkType
    training_algorithm: Optional[TrainingAlgorithm]
    action_space_type: ActionSpaceType
    version: str
    llm_config: Optional[LLMConfig]

class ModelMetadataHandler:
    """Class to handle loading and processing model metadata"""
    
    def __init__(self):
        self.metadata = None
    
    def load_model_metadata(self, file_path: str = "model_metadata.json") -> ModelMetadata:
        """
        Load model metadata from the specified JSON file
        
        Args:
            file_path: Path to the model metadata JSON file
            
        Returns:
            The loaded model metadata
            
        Raises:
            FileNotFoundError: If the model metadata file is not found
            json.JSONDecodeError: If the file contains invalid JSON
            ValueError: If the metadata validation fails
        """
        try:
            # Resolve path relative to current working directory
            resolved_path = os.path.abspath(file_path)
            logger.debug(f"Loading model metadata from: {resolved_path}")
            
            if not os.path.exists(resolved_path):
                raise FileNotFoundError(f"Model metadata file not found: {resolved_path}")
            
            with open(resolved_path, 'r') as file:
                self.metadata = json.load(file)
            
            # Validate the loaded metadata
            self._validate_metadata()
            
            logger.debug(
                f"Loaded model metadata with {self.get_action_space_type()} action space and "
                f"{self.metadata['neural_network']} neural network type"
            )
            
            return self.metadata
        
        except Exception as error:
            logger.error(f"Failed to load model metadata: {error}")
            raise
    
    def get_metadata(self) -> Optional[ModelMetadata]:
        """
        Get the current model metadata
        
        Returns:
            The model metadata or None if not loaded
        """
        return self.metadata
    
    def _validate_metadata(self) -> None:
        """
        Validate the loaded metadata structure
        
        Raises:
            ValueError: If the metadata is invalid
        """
        if not self.metadata:
            raise ValueError("No metadata loaded")
        
        # Check required fields
        if "action_space" not in self.metadata:
            raise ValueError("Missing action_space in model metadata")
        
        if "sensor" not in self.metadata or not isinstance(self.metadata["sensor"], list):
            raise ValueError("Missing or invalid sensor configuration in model metadata")
        
        if "neural_network" not in self.metadata:
            raise ValueError("Missing neural_network in model metadata")
        
        # Check for LLM configuration when neural network is LLM
        if self.metadata["neural_network"] == NeuralNetworkType.LLM.value:
            if "llm_config" not in self.metadata:
                raise ValueError("LLM neural network type requires llm_config to be specified")
            
            llm_config = self.metadata["llm_config"]
            if "model_id" not in llm_config:
                raise ValueError("Missing model_id in llm_config")
            
            if not isinstance(llm_config.get("max_tokens"), int):
                raise ValueError("Missing or invalid max_tokens in llm_config")
            
            if not isinstance(llm_config.get("system_prompt"), (str, list)):
                raise ValueError("Missing or invalid system_prompt in llm_config")
            
            if not isinstance(llm_config.get("context_window"), int) or llm_config.get("context_window", 0) < 0:
                raise ValueError("Invalid context_window in llm_config")
        else:
            # Traditional neural networks need training algorithm
            if "training_algorithm" not in self.metadata:
                raise ValueError("Missing training_algorithm for traditional neural network")
        
        # Validate action space based on type
        action_space_type = self.get_action_space_type()
        
        if action_space_type == ActionSpaceType.DISCRETE:
            # For discrete, action_space should be a list of actions
            if not isinstance(self.metadata["action_space"], list):
                raise ValueError("Discrete action space should be a list")
            
            # Validate each action
            for action in self.metadata["action_space"]:
                if not isinstance(action.get("steering_angle"), (int, float)) or \
                   not isinstance(action.get("speed"), (int, float)):
                    raise ValueError("Invalid discrete action format")
        
        elif action_space_type == ActionSpaceType.CONTINUOUS:
            # For continuous, action_space should be an object with ranges
            continuous_space = self.metadata["action_space"]
            
            if "speed" not in continuous_space or "steering_angle" not in continuous_space:
                raise ValueError("Continuous action space missing required ranges")
            
            if not isinstance(continuous_space["speed"].get("low"), (int, float)) or \
               not isinstance(continuous_space["speed"].get("high"), (int, float)) or \
               not isinstance(continuous_space["steering_angle"].get("low"), (int, float)) or \
               not isinstance(continuous_space["steering_angle"].get("high"), (int, float)):
                raise ValueError("Invalid continuous action range format")
            
            # Validate ranges
            if continuous_space["speed"]["low"] >= continuous_space["speed"]["high"]:
                raise ValueError("Speed range low must be less than high")
            
            if continuous_space["steering_angle"]["low"] >= continuous_space["steering_angle"]["high"]:
                raise ValueError("Steering angle range low must be less than high")
    
    def get_action_space_type(self) -> ActionSpaceType:
        """
        Get the action space type from the metadata
        
        Returns:
            ActionSpaceType enum value
            
        Raises:
            ValueError: If no metadata is loaded
        """
        if not self.metadata:
            raise ValueError("No metadata loaded")
        
        # Check if action_space_type is explicitly specified
        if "action_space_type" in self.metadata:
            return ActionSpaceType(self.metadata["action_space_type"])
        
        # Otherwise infer from action_space structure
        return (ActionSpaceType.DISCRETE if isinstance(self.metadata["action_space"], list)
                else ActionSpaceType.CONTINUOUS)
    
    def get_action_space(self) -> ActionSpace:
        """
        Get the action space configuration
        
        Returns:
            Action space configuration
            
        Raises:
            ValueError: If no metadata is loaded
        """
        if not self.metadata:
            raise ValueError("No metadata loaded")
        
        return self.metadata["action_space"]
    
    def is_llm_model(self) -> bool:
        """
        Check if the model uses an LLM
        
        Returns:
            True if the model uses an LLM, False otherwise
            
        Raises:
            ValueError: If no metadata is loaded
        """
        if not self.metadata:
            raise ValueError("No metadata loaded")
        
        return self.metadata["neural_network"] == NeuralNetworkType.LLM.value
    
    def get_llm_config(self) -> Optional[LLMConfig]:
        """
        Get the LLM configuration
        
        Returns:
            LLM configuration or None if not an LLM model
            
        Raises:
            ValueError: If no metadata is loaded
        """
        if not self.metadata or not self.is_llm_model():
            return None
        
        return self.metadata.get("llm_config")
    
    def get_discrete_actions(self) -> Optional[List[DiscreteAction]]:
        """
        Get discrete actions from metadata
        
        Returns:
            Array of discrete actions or None if action space is not discrete
            
        Raises:
            ValueError: If no metadata is loaded
        """
        if not self.metadata or self.get_action_space_type() != ActionSpaceType.DISCRETE:
            return None
        
        return self.metadata["action_space"]
    
    def get_continuous_action_space(self) -> Optional[ContinuousActionSpace]:
        """
        Get continuous action space ranges
        
        Returns:
            Continuous action space ranges or None if action space is not continuous
            
        Raises:
            ValueError: If no metadata is loaded
        """
        if not self.metadata or self.get_action_space_type() != ActionSpaceType.CONTINUOUS:
            return None
        
        return self.metadata["action_space"]
    
    def is_valid_steering_angle(self, steering_angle: float) -> bool:
        """
        Check if a steering angle is within the valid range
        
        Args:
            steering_angle: The steering angle to check
            
        Returns:
            Whether the steering angle is valid
            
        Raises:
            ValueError: If no metadata is loaded
        """
        if not self.metadata:
            raise ValueError("No metadata loaded")
        
        if self.get_action_space_type() == ActionSpaceType.CONTINUOUS:
            steering_range = self.metadata["action_space"]["steering_angle"]
            return steering_angle >= steering_range["low"] and steering_angle <= steering_range["high"]
        else:
            actions = self.metadata["action_space"]
            return any(action["steering_angle"] == steering_angle for action in actions)
    
    def is_valid_speed(self, speed: float) -> bool:
        """
        Check if a speed value is within the valid range
        
        Args:
            speed: The speed to check
            
        Returns:
            Whether the speed is valid
            
        Raises:
            ValueError: If no metadata is loaded
        """
        if not self.metadata:
            raise ValueError("No metadata loaded")
        
        if self.get_action_space_type() == ActionSpaceType.CONTINUOUS:
            speed_range = self.metadata["action_space"]["speed"]
            return speed >= speed_range["low"] and speed <= speed_range["high"]
        else:
            actions = self.metadata["action_space"]
            return any(action["speed"] == speed for action in actions)
    
    def get_neural_network_type(self) -> NeuralNetworkType:
        """
        Get the neural network type from metadata
        
        Returns:
            Neural network type
            
        Raises:
            ValueError: If no metadata is loaded
        """
        if not self.metadata:
            raise ValueError("No metadata loaded")
        
        return NeuralNetworkType(self.metadata["neural_network"])
    
    def get_training_algorithm(self) -> Optional[TrainingAlgorithm]:
        """
        Get the training algorithm from metadata
        
        Returns:
            Training algorithm or None if using LLM
            
        Raises:
            ValueError: If no metadata is loaded
        """
        if not self.metadata:
            raise ValueError("No metadata loaded")
        
        if "training_algorithm" not in self.metadata:
            return None
            
        return TrainingAlgorithm(self.metadata["training_algorithm"])
    
    def get_sensors(self) -> List[SensorType]:
        """
        Get the sensors configuration from metadata
        
        Returns:
            List of sensors
            
        Raises:
            ValueError: If no metadata is loaded
        """
        if not self.metadata:
            raise ValueError("No metadata loaded")
        
        return [SensorType(sensor) for sensor in self.metadata["sensor"]]
    
    def find_closest_discrete_action(self, steering_angle: float, speed: float) -> Optional[DiscreteAction]:
        """
        Find the closest discrete action to the given continuous values
        Only applicable for discrete action spaces
        
        Args:
            steering_angle: Target steering angle
            speed: Target speed
            
        Returns:
            The closest discrete action or None if not applicable
        """
        if not self.metadata or self.get_action_space_type() != ActionSpaceType.DISCRETE:
            return None
        
        actions = self.metadata["action_space"]
        
        closest_action = actions[0]
        min_distance = float('inf')
        
        for action in actions:
            # Calculate Euclidean distance in the action space
            steering_diff = action["steering_angle"] - steering_angle
            speed_diff = action["speed"] - speed
            distance = math.sqrt(steering_diff * steering_diff + speed_diff * speed_diff)
            
            if distance < min_distance:
                min_distance = distance
                closest_action = action
        
        return closest_action
    
    def normalize_action(self, steering_angle: float, speed: float) -> DiscreteAction:
        """
        Convert a continuous action to a valid discrete action if needed
        
        Args:
            steering_angle: Steering angle value
            speed: Speed value
            
        Returns:
            Valid action based on the action space type
            
        Raises:
            ValueError: If no metadata is loaded
        """
        if not self.metadata:
            raise ValueError("No metadata loaded")
        
        if self.get_action_space_type() == ActionSpaceType.CONTINUOUS:
            continuous_space = self.metadata["action_space"]
            
            # Check if values exceed the valid ranges and log warnings
            if (steering_angle < continuous_space["steering_angle"]["low"] or 
                steering_angle > continuous_space["steering_angle"]["high"]):
                logger.warning(
                    f"Steering angle {steering_angle} exceeds valid range "
                    f"[{continuous_space['steering_angle']['low']}, {continuous_space['steering_angle']['high']}]"
                )
            
            if (speed < continuous_space["speed"]["low"] or 
                speed > continuous_space["speed"]["high"]):
                logger.warning(
                    f"Speed {speed} exceeds valid range "
                    f"[{continuous_space['speed']['low']}, {continuous_space['speed']['high']}]"
                )
            
            # Clamp values to the valid ranges
            normalized_steering_angle = max(
                continuous_space["steering_angle"]["low"],
                min(continuous_space["steering_angle"]["high"], steering_angle)
            )
            
            normalized_speed = max(
                continuous_space["speed"]["low"],
                min(continuous_space["speed"]["high"], speed)
            )
            
            return {
                "steering_angle": normalized_steering_angle,
                "speed": normalized_speed
            }
        else:
            # For discrete action spaces, log a warning if the requested action is not exact
            discrete_actions = self.metadata["action_space"]
            exact_match = any(
                action["steering_angle"] == steering_angle and action["speed"] == speed
                for action in discrete_actions
            )
            
            if not exact_match:
                logger.warning(
                    f"Requested action ({steering_angle}, {speed}) is not in the discrete action space, "
                    f"finding closest match"
                )
            
            # Find closest discrete action
            closest_action = self.find_closest_discrete_action(steering_angle, speed)
            if not closest_action:
                raise ValueError("Failed to find a valid discrete action")
            
            return closest_action


# Create a singleton instance for convenience
model_metadata = ModelMetadataHandler()
