import "dotenv/config";
import * as path from "path";
import * as fs from "fs";
import BedrockService from "./services/bedrock";
import PricingService, { TokenPricing } from "./services/pricing";
import { Logger, LogLevel, getLogger } from "./utils/logger";

/**
 * DeepRacer LLM Agent to process track images and make driving decisions
 */
class DeepRacerAgent {
  private bedrockService: BedrockService;
  private pricingService: PricingService;
  private modelId: string;
  private imageCount: number = 0;
  private totalPromptTokens: number = 0;
  private totalCompletionTokens: number = 0;
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
    this.bedrockService = new BedrockService();

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
    if (options.systemPrompt) {
      this.bedrockService.setSystemPrompt(options.systemPrompt);
    } else {
      this.bedrockService.setSystemPrompt(
        `You are an AI driver assistant acting like a Rally navigator for an AWS DeepRacer 1/18th scale car. ` +
          `Your job is to analyze track images from the car's perspective and suggest optimal actions. ` +
          `You should consider the track features, curves, and obstacles to make driving decisions. ` +
          `The car is a 1/18th scale model, so the speed and steering angles should be realistic for a small car. ` +
          `The car has ackermann steering geometry, so the steering angle should be between -20 and +20 degrees. ` +
          `Full steering gives a turning radius of 0.5m, and the car can go up to 4 m/s. ` +
          `Always provide output in JSON format with "speed" (1-4 m/s) and "steering_angle" (-20 to +20 degrees) as floats. ` +
          `Positive steering angles turn the car left, negative steering angles turn the car right. ` +
          `Do not add + before any positive steering angle. ` +
          `Include short "reasoning" in your response to explain your decision. ` +
          `Include a field cotaining your cumulative "knowledge", showing what you have learned about driving the car. Retain knowledge from previous iterations.` +
          ``
      );
    }

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

    let prompt = "Analyze this image from the DeepRacer car's camera.";
    if (this.imageCount > 1) {
      prompt += " Compare with previous image to interpret how you are moving.";
    } else {
      prompt +=
        " This is the first image. The first curve is slightly to the left.";
    }

    try {
      const response = await this.bedrockService.processImageSync(
        imageBuffer,
        this.modelId,
        prompt
      );

      // Track token usage
      this.trackTokenUsage(response);

      // Extract and parse JSON from the response
      let drivingAction: any;

      try {
        if (this.modelId.includes("claude")) {
          const content = response.content?.[0]?.text || "";
          // Extract JSON from content - Claude often wraps it in ```json blocks
          const jsonMatch = content.match(
            /```json\s*(\{.*?\})\s*```|(\{.*?\})/s
          );
          if (jsonMatch) {
            const jsonString = jsonMatch[1] || jsonMatch[2];
            drivingAction = JSON.parse(jsonString.trim());
          } else {
            throw new Error("No JSON found in response");
          }
        } else if (this.modelId.includes("mistral")) {
          // Direct access to response for Mistral models
          if (response.choices && response.choices[0]?.message?.content) {
            const content = response.choices[0].message.content;

            // Log the raw content for debugging
            this.logger.debug("Raw content from Mistral model:", content);

            // Try to extract JSON from content if it's wrapped in code blocks
            // Using a more robust regex pattern
            const jsonMatch = content.match(
              /```(?:json)?\s*([\s\S]*?)\s*```|(\{[\s\S]*?\})/
            );
            if (jsonMatch) {
              const jsonString = (jsonMatch[1] || jsonMatch[2]).trim();
              this.logger.debug("Extracted JSON string:", jsonString);

              try {
                drivingAction = JSON.parse(jsonString);
              } catch (jsonParseError) {
                this.logger.error(
                  "Failed to parse extracted JSON:",
                  jsonParseError
                );
                throw new Error("Invalid JSON in response");
              }
            } else {
              // If no code block found, try parsing the entire content
              try {
                drivingAction = JSON.parse(content.trim());
                this.logger.debug("Parsed entire content as JSON");
              } catch (e) {
                this.logger.error("Failed to parse content as JSON:", content);
                throw new Error("No valid JSON found in response");
              }
            }
          } else {
            this.logger.error(
              "Unexpected response structure:",
              JSON.stringify(response)
            );
            throw new Error("Unexpected Mistral response structure");
          }
        } else if (this.modelId.includes("amazon.nova")) {
          // Handle Amazon Nova response format
          if (response.output?.message?.content) {
            const content = response.output.message.content[0]?.text || "";
            this.logger.debug("Raw content from Nova model:", content);

            // Try to extract JSON from content if it's wrapped in code blocks
            const jsonMatch = content.match(
              /```(?:json)?\s*([\s\S]*?)\s*```|(\{[\s\S]*?\})/
            );

            if (jsonMatch) {
              const jsonString = (jsonMatch[1] || jsonMatch[2]).trim();
              this.logger.debug("Extracted JSON string:", jsonString);

              try {
                drivingAction = JSON.parse(jsonString);
              } catch (jsonParseError) {
                this.logger.error(
                  "Failed to parse extracted JSON from Nova response:",
                  jsonParseError
                );
                throw new Error("Invalid JSON in Nova response");
              }
            } else {
              // If no code block found, try parsing the entire content
              try {
                drivingAction = JSON.parse(content.trim());
                this.logger.debug("Parsed entire Nova content as JSON");
              } catch (e) {
                this.logger.error(
                  "Failed to parse Nova content as JSON:",
                  content
                );
                throw new Error("No valid JSON found in Nova response");
              }
            }
          } else {
            this.logger.error(
              "Unexpected Nova response structure:",
              JSON.stringify(response)
            );
            throw new Error("Unexpected Nova response structure");
          }
        } else {
          // For other models, try to parse the entire response as JSON
          drivingAction = response;
        }
        // Validate the driving action
        if (
          drivingAction.speed === undefined ||
          drivingAction.steering_angle === undefined
        ) {
          this.logger.warn("Missing required driving parameters in response:");
          this.logger.warn("Raw response:", JSON.stringify(response, null, 2));

          // You could also provide default values for missing parameters
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

        // Last resort attempt - try to find any JSON in the full response
        try {
          const fullResponseStr = JSON.stringify(response);
          // Look for any JSON object with speed and steering_angle
          const jsonMatch = fullResponseStr.match(/"content"\s*:\s*"(.*?)",/);
          if (jsonMatch) {
            // Clean up escaped quotes and newlines
            const content = jsonMatch[1]
              .replace(/\\"/g, '"')
              .replace(/\\n/g, "\n");

            // Look for JSON in the content
            const jsonObjectMatch = content.match(
              /```json\s*(\{.*?\})\s*```|(\{.*?\})/s
            );
            if (jsonObjectMatch) {
              const jsonString = (jsonObjectMatch[1] || jsonObjectMatch[2])
                .replace(/\\n/g, "\n")
                .replace(/\\\\/g, "\\");
              return JSON.parse(jsonString.trim());
            }
          }
        } catch (lastResortError) {
          this.logger.error("Last resort parsing also failed");
        }

        throw new Error("Failed to parse driving action from LLM response");
      }
    } catch (error) {
      this.logger.error("Error in processImage:", error);
      throw error;
    }
  }

  /**
   * Track token usage from API responses
   * @param response The response from the API call
   */
  private trackTokenUsage(response: any): void {
    if (!response) return;

    try {
      // Different models return token counts in different formats
      if (response.usage) {
        if (response.usage.inputTokens !== undefined) {
          // Amazon Nova format
          const promptTokens = response.usage.inputTokens || 0;
          const completionTokens = response.usage.outputTokens || 0;

          this.totalPromptTokens += promptTokens;
          this.totalCompletionTokens += completionTokens;

          this.logger.debug(
            `Tokens - Prompt: ${promptTokens}, Completion: ${completionTokens}, Total: ${
              promptTokens + completionTokens
            }`
          );
        } else {
          // Mistral/Pixtral format
          const promptTokens = response.usage.prompt_tokens || 0;
          const completionTokens = response.usage.completion_tokens || 0;

          this.totalPromptTokens += promptTokens;
          this.totalCompletionTokens += completionTokens;

          this.logger.debug(
            `Tokens - Prompt: ${promptTokens}, Completion: ${completionTokens}, Total: ${
              promptTokens + completionTokens
            }`
          );
        }
      } else if (response.usage_metadata) {
        // Claude format
        const promptTokens = response.usage_metadata.input_tokens || 0;
        const completionTokens = response.usage_metadata.output_tokens || 0;

        this.totalPromptTokens += promptTokens;
        this.totalCompletionTokens += completionTokens;

        this.logger.debug(
          `Tokens - Prompt: ${promptTokens}, Completion: ${completionTokens}, Total: ${
            promptTokens + completionTokens
          }`
        );
      } else {
        this.logger.debug(`Token usage data not available for this response`);
      }
    } catch (error) {
      this.logger.warn("Error tracking token usage:", error);
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
    // Get current pricing
    const currentPricing = this.pricingService.getPricing();

    // Calculate costs using the pricing service
    const costs = this.pricingService.calculateCost(
      this.totalPromptTokens,
      this.totalCompletionTokens
    );

    return {
      promptTokens: this.totalPromptTokens,
      completionTokens: this.totalCompletionTokens,
      totalTokens: this.totalPromptTokens + this.totalCompletionTokens,
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
      this.totalPromptTokens = 0;
      this.totalCompletionTokens = 0;
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

// Example usage
async function main() {
  const mainLogger = getLogger("Main");

  // Parse command line arguments
  const args = process.argv.slice(2);
  const options: {
    frames?: number;
    skipFactor?: number;
    startOffset?: number;
    maxContextMessages?: number;
  } = {};

  // Process command line arguments
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--frames" || args[i] === "-f") {
      options.frames = parseInt(args[++i], 10);
      if (isNaN(options.frames) || options.frames <= 0) {
        mainLogger.error(
          "Invalid value for --frames. Must be a positive integer."
        );
        return;
      }
    } else if (args[i] === "--speed" || args[i] === "-x") {
      options.skipFactor = parseInt(args[++i], 10);
      if (isNaN(options.skipFactor) || options.skipFactor <= 0) {
        mainLogger.error(
          "Invalid value for --skip. Must be a positive integer."
        );
        return;
      }
    } else if (args[i] === "--start" || args[i] === "-s") {
      options.startOffset = parseInt(args[++i], 10);
      if (isNaN(options.startOffset) || options.startOffset < 0) {
        mainLogger.error(
          "Invalid value for --start. Must be a non-negative integer."
        );
        return;
      }
    } else if (args[i] === "--context" || args[i] === "-c") {
      options.maxContextMessages = parseInt(args[++i], 10);
      if (isNaN(options.maxContextMessages) || options.maxContextMessages < 0) {
        mainLogger.error(
          "Invalid value for --context. Must be a non-negative integer."
        );
        return;
      }
    } else if (args[i] === "--help" || args[i] === "-h") {
      console.log(`
DeepRacer LLM Agent Image Processing

Options:
  --frames, -f <number>   Number of frames to process (default: process all frames)
  --speed, -x <number>    Process every Nth frame (default: 2)
  --start, -s <number>    Start processing from Nth image (default: 0)
  --context, -c <number>  Maximum number of previous messages to retain in context (default: unlimited)
  --help, -h              Show this help message
      `);
      return;
    }
  }

  // Default skipFactor to 2 if not specified
  const skipFactor = options.skipFactor || 2;
  mainLogger.info(`Using frame skip factor: ${skipFactor}`);

  // Create the DeepRacer agent with a specified log level
  const logLevel = process.env.LOG_LEVEL
    ? LogLevel[process.env.LOG_LEVEL.toUpperCase() as keyof typeof LogLevel] ||
      LogLevel.INFO
    : LogLevel.INFO;
  const agent = new DeepRacerAgent({
    logLevel,
    maxContextMessages: options.maxContextMessages,
  });

  try {
    const testImagesDir = path.join(__dirname, "..", "test-images");

    if (!fs.existsSync(testImagesDir)) {
      mainLogger.error(`Test images directory not found: ${testImagesDir}`);
      return;
    }

    // Get all image files (jpg, jpeg, png) and sort them numerically
    const imageFiles = fs
      .readdirSync(testImagesDir)
      .filter((file) => /\.(jpg|jpeg|png)$/i.test(file))
      .sort((a, b) => {
        // Extract numbers from filenames for proper numeric sorting
        const numA = parseInt((a.match(/\d+/) || ["0"])[0]);
        const numB = parseInt((b.match(/\d+/) || ["0"])[0]);
        return numA - numB;
      });

    if (imageFiles.length === 0) {
      mainLogger.error("No image files found in test-images directory");
      return;
    }

    mainLogger.info(`Found ${imageFiles.length} images to process`);

    // Apply start offset if specified
    const startOffset = options.startOffset || 0;
    if (startOffset > 0) {
      mainLogger.info(
        `Starting from image ${startOffset} (skipping ${startOffset} images)`
      );
    }

    // Determine how many frames to process
    const maxFrames =
      options.frames ||
      Math.floor((imageFiles.length - startOffset) / skipFactor);
    const framesToProcess = Math.min(
      maxFrames,
      Math.floor((imageFiles.length - startOffset) / skipFactor)
    );
    mainLogger.info(
      `Will process ${framesToProcess} frames (every ${skipFactor}th frame)`
    );

    // Process each image in sequence with the specified skip factor
    for (let i = 0; i < framesToProcess; i++) {
      const frameIndex = startOffset + i * skipFactor;
      const imagePath = path.join(testImagesDir, imageFiles[frameIndex]);
      mainLogger.info(
        `\n[${i + 1}/${framesToProcess}] 🏎️ Processing image: ${
          imageFiles[frameIndex]
        }`
      );

      const action = await agent.processImageFile(imagePath);
      mainLogger.info("Recommended action:", JSON.stringify(action, null, 2));

      // Optional: Add a small delay between processing to avoid rate limits
      if (i < framesToProcess - 1) {
        mainLogger.debug("Waiting before processing next image...");
        await new Promise((resolve) => setTimeout(resolve, 50));
      }
    }

    // Log the total token usage with accurate pricing
    const tokenUsage = agent.getTokenUsage();
    mainLogger.info("\n📈 Token Usage Summary:");
    mainLogger.info(
      `   Prompt tokens:     ${tokenUsage.promptTokens.toLocaleString()}`
    );
    mainLogger.info(
      `   Completion tokens: ${tokenUsage.completionTokens.toLocaleString()}`
    );
    mainLogger.info(
      `   Total tokens:      ${tokenUsage.totalTokens.toLocaleString()}`
    );

    // Display pricing rates used for calculation
    mainLogger.info(
      `   Prompt rate:       $${tokenUsage.pricing.promptRate.toFixed(
        4
      )}/1K tokens`
    );
    mainLogger.info(
      `   Completion rate:   $${tokenUsage.pricing.completionRate.toFixed(
        4
      )}/1K tokens`
    );
    mainLogger.info(
      `   Estimated cost:    $${tokenUsage.estimatedCost.toFixed(4)}`
    );

    mainLogger.info("\n✅ All images processed successfully");
  } catch (error) {
    mainLogger.error("❌ Error processing images:", error);
  }
}

// Run the example if this script is executed directly
if (require.main === module) {
  main().catch(console.error);
}

// Export the DeepRacer agent and reexport LogLevel for convenience
export { LogLevel };
export default DeepRacerAgent;
