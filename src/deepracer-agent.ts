import "dotenv/config";
import * as path from "path";
import * as fs from "fs";
import BedrockService from "./services/bedrock";
import PricingService, { TokenPricing } from "./services/pricing";
import { Logger, LogLevel, getLogger } from "./utils/logger";
import { ModelFactory } from "./services/bedrock/models/model-factory";

/**
 * DeepRacer LLM Agent to process track images and make driving decisions
 */
class DeepRacerAgent {
  private bedrockService: BedrockService;
  private pricingService: PricingService;
  private modelId: string;
  private imageCount: number = 0;
  private logger: Logger;
  private maxContextMessages: number;

  constructor(
    options: {
      modelId?: string;
      systemPrompt?: string;
      logLevel?: LogLevel;
      maxContextMessages?: number;
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

    // Set model ID from options, env, or default
    this.modelId =
      options.modelId ||
      process.env.INFERENCE_PROFILE_ARN ||
      process.env.DEFAULT_MODEL_ID ||
      "";

    // Set maximum context messages to keep
    this.maxContextMessages =
      options.maxContextMessages ||
      (process.env.MAX_CONTEXT_MESSAGES
        ? parseInt(process.env.MAX_CONTEXT_MESSAGES)
        : 0);

    if (this.maxContextMessages > 0) {
      this.logger.info(
        `Context memory limited to last ${this.maxContextMessages} messages`
      );
      this.bedrockService.setMaxContextMessages(this.maxContextMessages);
    }

    // Set system prompt if provided
    const systemPrompt = options.systemPrompt || 
      `You are an AI driver assistant acting like a Rally navigator for an AWS DeepRacer 1/18th scale car. ` +
      `Your job is to analyze pictures looking at the track, looking forward out the window of the car. ` +
      `You should consider the track features, curves both near and far, to make driving decisions. ` +
      `The car has an Ackermann steering geometry. The steering angle should be between -20 and +20 degrees. ` +
      `IMPORTANT STEERING CONVENTION: Positive steering angles (+1 to +20) turn the car LEFT, negative steering angles (-1 to -20) turn the car RIGHT. ` +
      `Always provide output in JSON format with "speed" (1-4 m/s) and "steering_angle" (-20 to +20 degrees) as floats. Do not add + before any positive steering angle. ` + 
      `The track is having white lines to the left and the right, and a dashed yellow centerline. ` +
      `Include short "reasoning" in your response to explain your decision. ` +
      `Include a field containing your current "knowledge", structuring what you have learned about driving the car. Review and update knowledge from previous iterations.`;
    
    this.bedrockService.setSystemPrompt(systemPrompt);

    this.logger.info(
      `🚗 DeepRacer LLM Agent initialized with model: ${this.modelId}`
    );

    // Load pricing data for the model
    this.pricingService
      .loadModelPricing(this.modelId)
      .catch((error) =>
        this.logger.warn(`Failed to initialize pricing: ${error}`)
      );
  }

  /**
   * Process a new camera image and determine the next driving action
   * @param imageBuffer The camera image buffer
   * @returns Promise resolving to a recommended driving action
   */
  async processImage(imageBuffer: Buffer): Promise<any> {
    this.imageCount++;
    this.logger.debug(`Processing image #${this.imageCount}...`);

    let prompt = `Analyze this image. This is image #${this.imageCount}.`;
    if (this.imageCount > 1 && this.maxContextMessages > 0) {
      prompt += ` Compare with previous image to interpret how you are moving.`;
    }

    try {
      const response = await this.bedrockService.processImageSync(
        imageBuffer,
        this.modelId,
        prompt
      );

      // Token usage is now tracked inside the BedrockService
      // No need to explicitly track it here

      try {
        // Extract driving action using model-specific handler
        const drivingAction = this.bedrockService.extractDrivingAction(response);
        
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

        return drivingAction;
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

    if (refreshPricing) {
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
