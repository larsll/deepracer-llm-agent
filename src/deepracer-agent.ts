import "dotenv/config";
import * as fs from "fs";
import BedrockService from "./services/bedrock";
import PricingService, { TokenPricing } from "./services/pricing";
import { Logger, LogLevel, getLogger } from "./utils/logger";
import {
  ModelMetadata,
  modelMetadata as metadataHandler,
  ActionSpaceType,
  LLMConfig,
} from "./utils/model-metadata";

/**
 * DeepRacer LLM Agent to process track images and make driving decisions
 */
class DeepRacerAgent {
  private bedrockService: BedrockService;
  private pricingService: PricingService;
  private modelId: string = "";
  private imageCount: number = 0;
  private logger: Logger;
  private maxContextMessages: number = 0;
  private metadata?: ModelMetadata;
  private actionSpaceType?: ActionSpaceType;
  private initialized: boolean = false;

  constructor(
    options: {
      metadataFilePath?: string;
      logLevel?: LogLevel;
    } = {}
  ) {
    // Initialize logger
    this.logger = getLogger("DeepRacer", options.logLevel);

    // Initialize the Bedrock service
    this.bedrockService = new BedrockService(options.logLevel);

    // Initialize pricing service with AWS region from environment
    this.pricingService = new PricingService(
      process.env.AWS_REGION || "us-east-1"
    );

    // Load model metadata
    const metadataPath = options.metadataFilePath || "";
    this.loadModelMetadata(metadataPath);
  }

  /**
   * Load model metadata and initialize the agent's configuration
   * @param filePath Path to the model metadata file
   */
  private loadModelMetadata(filePath: string) {
    try {
      // Load and validate model metadata
      this.metadata = metadataHandler.loadModelMetadata(filePath);
      console.log("Model metadata loaded successfully:", this.metadata);
      this.actionSpaceType = metadataHandler.getActionSpaceType();

      // Check if we're using an LLM model
      if (metadataHandler.isLLMModel()) {
        const llmConfig = metadataHandler.getLLMConfig();

        if (!llmConfig) {
          throw new Error("LLM configuration missing");
        }

        // Configure LLM parameters
        this.setupLLMConfiguration(llmConfig);

        this.logger.info(
          `🚗 DeepRacer LLM Agent initialized with model: ${this.modelId}`
        );

        // Load pricing data for the model
        this.pricingService
          .loadModelPricing(this.modelId || "")
          .catch((error) =>
            this.logger.warn(`Failed to initialize pricing: ${error}`)
          );
      } else {
        this.logger.info(
          `🚗 DeepRacer Agent initialized with neural network type: ${metadataHandler.getNeuralNetworkType()}`
        );
      }
    } catch (error) {
      this.logger.error(
        `Failed to initialize agent with model metadata: ${error}`
      );
      throw error;
    }
  }

  /**
   * Configure the agent with LLM settings from metadata
   * @param llmConfig The LLM configuration
   */
  private setupLLMConfiguration(llmConfig: LLMConfig): void {
    // Set model ID from LLM config, with fallback to environment variables
    this.modelId =
      llmConfig.model_id ||
      process.env.INFERENCE_PROFILE_ARN ||
      process.env.DEFAULT_MODEL_ID ||
      "";

    if (!this.modelId) {
      throw new Error("No model ID specified in LLM config or environment");
    }

    // Set maximum context messages
    this.maxContextMessages = llmConfig.context_window || 0;

    if (this.maxContextMessages > 0) {
      this.logger.info(
        `Context memory limited to last ${this.maxContextMessages} messages`
      );
      this.bedrockService.setMaxContextMessages(this.maxContextMessages);
    }

    // Set system prompt
    if (Array.isArray(llmConfig.system_prompt)) {
      this.bedrockService.setSystemPrompt(llmConfig.system_prompt.join("\n"));
    } else {
      this.bedrockService.setSystemPrompt(llmConfig.system_prompt);
    }
  }

  /**
   * Process a new camera image and determine the next driving action
   * @param imageBuffer The camera image buffer
   * @returns Promise resolving to a recommended driving action
   */
  async processImage(imageBuffer: Buffer): Promise<any> {
    // Check if we have loaded metadata
    if (!this.metadata) {
      throw new Error("Model metadata not loaded");
    }

    // Check if we're using an LLM model
    if (!metadataHandler.isLLMModel()) {
      throw new Error("This method only supports LLM models");
    }

    const llmConfig = metadataHandler.getLLMConfig();
    if (!llmConfig) {
      throw new Error("LLM configuration missing");
    }

    this.imageCount++;
    this.logger.debug(`Processing image #${this.imageCount}...`);

    // Use the repeated prompt from the LLM config
    let prompt =
      llmConfig.repeated_prompt ||
      `Analyze this image. This is image #${this.imageCount}.`;

    if (this.imageCount > 1 && this.maxContextMessages > 0) {
      prompt += ` Compare with previous image to interpret how you are moving.`;
    }

    try {
      const response = await this.bedrockService.processImageSync(
        imageBuffer,
        this.modelId,
        prompt
      );

      try {
        // Extract driving action using model-specific handler
        const drivingAction =
          this.bedrockService.extractDrivingAction(response);

        // Validate the driving action
        if (
          drivingAction.speed === undefined ||
          drivingAction.steering_angle === undefined
        ) {
          this.logger.warn("Missing required driving parameters in response:");
          this.logger.warn("Raw response:", JSON.stringify(response, null, 2));

          // Provide default values for missing parameters
          if (!drivingAction.speed) drivingAction.speed = 1.0; // Safe default speed
          if (!drivingAction.steering_angle) drivingAction.steering_angle = 0.0; // Neutral steering

          // Add a flag to indicate this was a fallback action
          drivingAction.fallback = true;
          drivingAction.error = "Missing required parameters in response";
        }

        // Normalize the action according to our action space limits
        return metadataHandler.normalizeAction(
          drivingAction.steering_angle,
          drivingAction.speed
        );
      } catch (parseError) {
        this.logger.error("Failed to parse driving action:", parseError);
        this.logger.debug("Raw response:", JSON.stringify(response, null, 2));
        throw new Error("Failed to parse driving action from LLM response");
      }
    } catch (error) {
      this.logger.error("Error in processImage:", error);
      throw error;
    }
  }

  /**
   * Get the current token usage statistics and cost estimation
   * @returns Object containing token usage data and cost estimation
   */
  public getTokenUsage(): {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
    estimatedCost: number;
    pricing: TokenPricing;
  } {
    // Get token usage directly from BedrockService instead of TokenLogger
    const tokenUsage = this.bedrockService.getTokenUsage();

    // Get current pricing
    const currentPricing = this.pricingService.getPricing();

    // Calculate costs using the pricing service
    const costs = this.pricingService.calculateCost(
      tokenUsage.promptTokens,
      tokenUsage.completionTokens
    );

    return {
      ...tokenUsage,
      estimatedCost: costs.totalCost,
      pricing: currentPricing,
    };
  }

  /**
   * Process an image file and determine the next driving action
   * @param filePath Path to the image file
   * @returns Promise resolving to a recommended driving action
   */
  async processImageFile(filePath: string): Promise<any> {
    if (!fs.existsSync(filePath)) {
      throw new Error(`Image file not found: ${filePath}`);
    }

    const imageBuffer = fs.readFileSync(filePath);
    return this.processImage(imageBuffer);
  }

  /**
   * Reset the agent's conversation history and token counts
   * @param resetTokens Whether to reset token tracking (default: false)
   * @param refreshPricing Whether to refresh pricing data (default: false)
   */
  reset(resetTokens: boolean = false, refreshPricing: boolean = false): void {
    this.bedrockService.clearConversation();
    this.imageCount = 0;

    if (resetTokens) {
      this.logger.info("🔄 DeepRacer agent reset (including token counts)");
    } else {
      this.logger.info("🔄 DeepRacer agent reset");
    }

    if (refreshPricing && metadataHandler.isLLMModel()) {
      this.pricingService
        .loadModelPricing(this.modelId)
        .catch((error) =>
          this.logger.warn(`Failed to refresh pricing: ${error}`)
        );
    }
  }

  /**
   * Get the currently loaded model metadata
   * @returns The model metadata object
   */
  public getModelMetadata(): ModelMetadata {
    if (!this.metadata) {
      throw new Error("Model metadata is not loaded");
    }
    return this.metadata;
  }

  /**
   * Get the action space type (discrete or continuous)
   * @returns ActionSpaceType enum value
   */
  public getActionSpaceType(): ActionSpaceType {
    if (!this.actionSpaceType) {
      throw new Error("Action space type is not initialized");
    }
    return this.actionSpaceType;
  }
}

// Export the DeepRacer agent and reexport LogLevel for convenience
export { LogLevel };
export default DeepRacerAgent;
