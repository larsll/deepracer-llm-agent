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

  constructor(
    options: {
      metadataFilePath?: string;
      logLevel?: LogLevel;
    } = {}
  ) {
    // Initialize logger
    this.logger = getLogger("DeepRacer", options.logLevel);

    // Initialize pricing service with AWS region from environment
    this.pricingService = new PricingService(
      process.env.AWS_REGION || "us-east-1"
    );

    // Load model metadata
    this.loadModelMetadata(options.metadataFilePath || "");

    // Initialize BedrockService with metadata in a single step
    this.bedrockService = new BedrockService({
      metadata: this.metadata,
      logLevel: options.logLevel,
    });

    this.logger.info(
      `🚗 DeepRacer LLM Agent initialized with model: ${this.modelId} in ${process.env.AWS_REGION || "us-east-1"}`
    );

    // Load pricing data for the model
    this.pricingService
      .loadModelPricing(this.modelId || "")
      .catch((error) =>
        this.logger.warn(`Failed to initialize pricing: ${error}`)
      );
  }

  /**
   * Load model metadata and initialize the agent's configuration
   * @param filePath Path to the model metadata file
   */
  private loadModelMetadata(filePath: string) {
    try {
      // Load and validate model metadata
      this.metadata = metadataHandler.loadModelMetadata(filePath);
      this.logger.debug("Model metadata loaded successfully:", this.metadata);

      // Check if we're using an LLM model
      if (metadataHandler.isLLMModel()) {
        if (!metadataHandler.getLLMConfig()) {
          throw new Error("LLM configuration missing");
        }

        // Extract the model ID for this agent
        this.setupAgentConfiguration(this.metadata);
      } else {
        throw new Error(`🚗 DeepRacer Agent only works with LLM models.`);
      }
    } catch (error) {
      this.logger.error(
        `Failed to initialize agent with model metadata: ${error}`
      );
      throw error;
    }
  }

  /**
   * Configure the agent with settings from metadata
   * @param metadata The model metadata
   */
  private setupAgentConfiguration(metadata: ModelMetadata): void {
    const llmConfig = metadata.llm_config;
    if (!llmConfig) {
      throw new Error("LLM configuration missing");
    }

    // Set model ID from LLM config, with fallback to environment variables
    this.modelId =
      llmConfig.model_id ||
      process.env.INFERENCE_PROFILE_ARN ||
      process.env.DEFAULT_MODEL_ID ||
      "";

    if (!this.modelId) {
      throw new Error("No model ID specified in LLM config or environment");
    }

    // Store maximum context messages for the agent's own reference
    this.maxContextMessages = llmConfig.context_window || 0;

    if (this.maxContextMessages > 0) {
      this.logger.debug(
        `Context memory limited to last ${this.maxContextMessages} messages`
      );
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
      const drivingAction = await this.bedrockService.processImageSync(
        imageBuffer,
        prompt
      );

      try {
        this.logger.info("Extracted driving action:", drivingAction);

        // Validate the driving action
        if (
          drivingAction.speed === undefined ||
          drivingAction.steering_angle === undefined
        ) {
          this.logger.warn("Missing required driving parameters in response:");
          this.logger.warn(
            "Raw response:",
            JSON.stringify(drivingAction, null, 2)
          );

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
}

// Export the DeepRacer agent and reexport LogLevel for convenience
export { LogLevel };
export default DeepRacerAgent;
