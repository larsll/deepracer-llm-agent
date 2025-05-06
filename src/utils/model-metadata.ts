import * as fs from "fs";
import * as path from "path";
import { Logger, getLogger } from "./logger";

/**
 * Interface for discrete action space configuration
 */
export interface DiscreteAction {
  steering_angle: number;
  speed: number;
}

/**
 * Interface for continuous action space range
 */
export interface ContinuousRange {
  low: number;
  high: number;
}

/**
 * Interface for continuous action space configuration
 */
export interface ContinuousActionSpace {
  speed: ContinuousRange;
  steering_angle: ContinuousRange;
}

/**
 * Type definition for action space which can be either discrete or continuous
 */
export type ActionSpace = DiscreteAction[] | ContinuousActionSpace;

/**
 * Action space type enumeration
 */
export enum ActionSpaceType {
  DISCRETE = "discrete",
  CONTINUOUS = "continuous",
}

/**
 * Neural network type enumeration
 */
export enum NeuralNetworkType {
  DEEP_CONVOLUTIONAL_NETWORK_SHALLOW = "DEEP_CONVOLUTIONAL_NETWORK_SHALLOW",
  DEEP_CONVOLUTIONAL_NETWORK = "DEEP_CONVOLUTIONAL_NETWORK",
  LLM = "LLM",
}

/**
 * Training algorithm type enumeration
 */
export enum TrainingAlgorithm {
  CLIPPED_PPO = "clipped_ppo",
  SAC = "sac",
}

/**
 * Sensor type enumeration
 */
export enum SensorType {
  FRONT_FACING_CAMERA = "FRONT_FACING_CAMERA",
  STEREO_CAMERAS = "STEREO_CAMERAS",
  LIDAR = "LIDAR",
}

/**
 * Interface for LLM configuration
 */
export interface LLMConfig {
  model_id: string;
  max_tokens: number;
  system_prompt: string;
  repeated_prompt: string;
  context_window: number;
}

/**
 * Interface for model metadata configuration
 */
export interface ModelMetadata {
  action_space: ActionSpace;
  sensor: SensorType[];
  neural_network: NeuralNetworkType;
  training_algorithm?: TrainingAlgorithm;
  action_space_type: ActionSpaceType;
  version: string;
  llm_config?: LLMConfig;
}

/**
 * Class to handle loading and processing model metadata
 */
export class ModelMetadataHandler {
  private logger: Logger;
  private metadata: ModelMetadata | null = null;

  constructor() {
    this.logger = getLogger("ModelMetadata");
  }

  /**
   * Load model metadata from the specified JSON file
   *
   * @param filePath Path to the model metadata JSON file
   * @returns Promise resolving to the loaded model metadata
   */
  public loadModelMetadata(
    filePath: string = "model_metadata.json"
  ): ModelMetadata {
    try {
      // Resolve path relative to project root
      const resolvedPath = path.resolve(process.cwd(), filePath);

      this.logger.debug(`Loading model metadata from: ${resolvedPath}`);

      if (!fs.existsSync(resolvedPath)) {
        throw new Error(`Model metadata file not found: ${resolvedPath}`);
      }

      const fileContent = fs.readFileSync(resolvedPath, "utf8");
      this.metadata = JSON.parse(fileContent) as ModelMetadata;

      // Validate the loaded metadata
      this.validateMetadata();

      this.logger.info(
        `Loaded model metadata with ${this.getActionSpaceType()} action space and ${
          this.metadata.neural_network
        } neural network type`
      );

      return this.metadata;
    } catch (error) {
      this.logger.error(`Failed to load model metadata: ${error}`);
      throw error;
    }
  }

  /**
   * Get the current model metadata
   * @returns The model metadata or null if not loaded
   */
  public getMetadata(): ModelMetadata | null {
    return this.metadata;
  }

  /**
   * Validate the loaded metadata structure
   */
  private validateMetadata(): void {
    if (!this.metadata) {
      throw new Error("No metadata loaded");
    }

    // Check required fields
    if (!this.metadata.action_space) {
      throw new Error("Missing action_space in model metadata");
    }

    if (!this.metadata.sensor || !Array.isArray(this.metadata.sensor)) {
      throw new Error(
        "Missing or invalid sensor configuration in model metadata"
      );
    }

    if (!this.metadata.neural_network) {
      throw new Error("Missing neural_network in model metadata");
    }

    // Check for LLM configuration when neural network is LLM
    if (this.metadata.neural_network === NeuralNetworkType.LLM) {
      if (!this.metadata.llm_config) {
        throw new Error(
          "LLM neural network type requires llm_config to be specified"
        );
      }

      const llmConfig = this.metadata.llm_config;
      if (!llmConfig.model_id) {
        throw new Error("Missing model_id in llm_config");
      }

      if (typeof llmConfig.max_tokens !== "number") {
        throw new Error("Missing or invalid max_tokens in llm_config");
      }

      if (typeof llmConfig.system_prompt !== "string") {
        throw new Error("Missing system_prompt in llm_config");
      }

      if (typeof llmConfig.repeated_prompt !== "string") {
        throw new Error("Missing repeated_prompt in llm_config");
      }

      if (
        typeof llmConfig.context_window !== "number" ||
        llmConfig.context_window < 0
      ) {
        throw new Error("Invalid context_window in llm_config");
      }
    } else {
      // Traditional neural networks need training algorithm
      if (!this.metadata.training_algorithm) {
        throw new Error(
          "Missing training_algorithm in model metadata for traditional neural network"
        );
      }
    }

    // Validate action space based on type
    const actionSpaceType = this.getActionSpaceType();

    if (actionSpaceType === ActionSpaceType.DISCRETE) {
      // For discrete, action_space should be an array of actions
      if (!Array.isArray(this.metadata.action_space)) {
        throw new Error("Discrete action space should be an array");
      }

      // Validate each action
      for (const action of this.metadata.action_space as DiscreteAction[]) {
        if (
          typeof action.steering_angle !== "number" ||
          typeof action.speed !== "number"
        ) {
          throw new Error("Invalid discrete action format");
        }
      }
    } else if (actionSpaceType === ActionSpaceType.CONTINUOUS) {
      // For continuous, action_space should be an object with ranges
      const continuousSpace = this.metadata
        .action_space as ContinuousActionSpace;

      if (!continuousSpace.speed || !continuousSpace.steering_angle) {
        throw new Error("Continuous action space missing required ranges");
      }

      if (
        typeof continuousSpace.speed.low !== "number" ||
        typeof continuousSpace.speed.high !== "number" ||
        typeof continuousSpace.steering_angle.low !== "number" ||
        typeof continuousSpace.steering_angle.high !== "number"
      ) {
        throw new Error("Invalid continuous action range format");
      }

      // Validate ranges
      if (continuousSpace.speed.low >= continuousSpace.speed.high) {
        throw new Error("Speed range low must be less than high");
      }

      if (
        continuousSpace.steering_angle.low >=
        continuousSpace.steering_angle.high
      ) {
        throw new Error("Steering angle range low must be less than high");
      }
    }
  }

  /**
   * Get the action space type from the metadata
   * @returns ActionSpaceType enum value
   */
  public getActionSpaceType(): ActionSpaceType {
    if (!this.metadata) {
      throw new Error("No metadata loaded");
    }

    return (
      this.metadata.action_space_type ||
      (Array.isArray(this.metadata.action_space)
        ? ActionSpaceType.DISCRETE
        : ActionSpaceType.CONTINUOUS)
    );
  }

  /**
   * Get the action space configuration
   * @returns Action space configuration
   */
  public getActionSpace(): ActionSpace {
    if (!this.metadata) {
      throw new Error("No metadata loaded");
    }

    return this.metadata.action_space;
  }

  /**
   * Check if the model uses an LLM
   * @returns True if the model uses an LLM, false otherwise
   */
  public isLLMModel(): boolean {
    if (!this.metadata) {
      throw new Error("No metadata loaded");
    }

    return this.metadata.neural_network === NeuralNetworkType.LLM;
  }

  /**
   * Get the LLM configuration
   * @returns LLM configuration or null if not an LLM model
   */
  public getLLMConfig(): LLMConfig | null {
    if (!this.metadata || !this.isLLMModel()) {
      return null;
    }

    return this.metadata.llm_config || null;
  }

  /**
   * Get discrete actions from metadata
   * @returns Array of discrete actions or null if action space is not discrete
   */
  public getDiscreteActions(): DiscreteAction[] | null {
    if (
      !this.metadata ||
      this.getActionSpaceType() !== ActionSpaceType.DISCRETE
    ) {
      return null;
    }

    return this.metadata.action_space as DiscreteAction[];
  }

  /**
   * Get continuous action space ranges
   * @returns Continuous action space ranges or null if action space is not continuous
   */
  public getContinuousActionSpace(): ContinuousActionSpace | null {
    if (
      !this.metadata ||
      this.getActionSpaceType() !== ActionSpaceType.CONTINUOUS
    ) {
      return null;
    }

    return this.metadata.action_space as ContinuousActionSpace;
  }

  /**
   * Check if a steering angle is within the valid range
   * @param steeringAngle The steering angle to check
   * @returns Whether the steering angle is valid
   */
  public isValidSteeringAngle(steeringAngle: number): boolean {
    if (!this.metadata) {
      throw new Error("No metadata loaded");
    }

    if (this.getActionSpaceType() === ActionSpaceType.CONTINUOUS) {
      const { steering_angle } = this.metadata
        .action_space as ContinuousActionSpace;
      return (
        steeringAngle >= steering_angle.low &&
        steeringAngle <= steering_angle.high
      );
    } else {
      const actions = this.metadata.action_space as DiscreteAction[];
      return actions.some((action) => action.steering_angle === steeringAngle);
    }
  }

  /**
   * Check if a speed value is within the valid range
   * @param speed The speed to check
   * @returns Whether the speed is valid
   */
  public isValidSpeed(speed: number): boolean {
    if (!this.metadata) {
      throw new Error("No metadata loaded");
    }

    if (this.getActionSpaceType() === ActionSpaceType.CONTINUOUS) {
      const { speed: speedRange } = this.metadata
        .action_space as ContinuousActionSpace;
      return speed >= speedRange.low && speed <= speedRange.high;
    } else {
      const actions = this.metadata.action_space as DiscreteAction[];
      return actions.some((action) => action.speed === speed);
    }
  }

  /**
   * Get the neural network type from metadata
   * @returns Neural network type
   */
  public getNeuralNetworkType(): NeuralNetworkType {
    if (!this.metadata) {
      throw new Error("No metadata loaded");
    }

    return this.metadata.neural_network;
  }

  /**
   * Get the training algorithm from metadata
   * @returns Training algorithm or undefined if using LLM
   */
  public getTrainingAlgorithm(): TrainingAlgorithm | undefined {
    if (!this.metadata) {
      throw new Error("No metadata loaded");
    }

    return this.metadata.training_algorithm;
  }

  /**
   * Get the sensors configuration from metadata
   * @returns Array of sensors
   */
  public getSensors(): SensorType[] {
    if (!this.metadata) {
      throw new Error("No metadata loaded");
    }

    return this.metadata.sensor;
  }

  /**
   * Find the closest discrete action to the given continuous values
   * Only applicable for discrete action spaces
   *
   * @param steeringAngle Target steering angle
   * @param speed Target speed
   * @returns The closest discrete action
   */
  public findClosestDiscreteAction(
    steeringAngle: number,
    speed: number
  ): DiscreteAction | null {
    if (
      !this.metadata ||
      this.getActionSpaceType() !== ActionSpaceType.DISCRETE
    ) {
      return null;
    }

    const actions = this.metadata.action_space as DiscreteAction[];

    let closestAction = actions[0];
    let minDistance = Infinity;

    for (const action of actions) {
      // Calculate Euclidean distance in the action space
      const steeringDiff = action.steering_angle - steeringAngle;
      const speedDiff = action.speed - speed;
      const distance = Math.sqrt(
        steeringDiff * steeringDiff + speedDiff * speedDiff
      );

      if (distance < minDistance) {
        minDistance = distance;
        closestAction = action;
      }
    }

    return closestAction;
  }

  /**
   * Convert a continuous action to a valid discrete action if needed
   *
   * @param steeringAngle Steering angle value
   * @param speed Speed value
   * @returns Valid action based on the action space type
   */
  public normalizeAction(steeringAngle: number, speed: number): DiscreteAction {
    if (!this.metadata) {
      throw new Error("No metadata loaded");
    }

    if (this.getActionSpaceType() === ActionSpaceType.CONTINUOUS) {
      const continuousSpace = this.metadata
        .action_space as ContinuousActionSpace;

      // Clamp values to the valid ranges
      const normalizedSteeringAngle = Math.max(
        continuousSpace.steering_angle.low,
        Math.min(continuousSpace.steering_angle.high, steeringAngle)
      );

      const normalizedSpeed = Math.max(
        continuousSpace.speed.low,
        Math.min(continuousSpace.speed.high, speed)
      );

      return {
        steering_angle: normalizedSteeringAngle,
        speed: normalizedSpeed,
      };
    } else {
      // Find closest discrete action
      const closestAction = this.findClosestDiscreteAction(
        steeringAngle,
        speed
      );
      if (!closestAction) {
        throw new Error("Failed to find a valid discrete action");
      }
      return closestAction;
    }
  }
}

// Export a singleton instance for convenience
export const modelMetadata = new ModelMetadataHandler();

// Default export for backwards compatibility
export default modelMetadata;
