import "dotenv/config";
import * as path from "path";
import * as fs from "fs";
import BedrockService from "./services/bedrock";
import { Logger, LogLevel, getLogger } from "./utils/logger";

/**
 * DeepRacer LLM Agent to process track images and make driving decisions
 */
class DeepRacerAgent {
  private bedrockService: BedrockService;
  private modelId: string;
  private imageCount: number = 0;
  private totalPromptTokens: number = 0;
  private totalCompletionTokens: number = 0;
  private logger: Logger;

  constructor(
    options: {
      modelId?: string;
      systemPrompt?: string;
      logLevel?: LogLevel;
    } = {}
  ) {
    // Initialize logger
    this.logger = getLogger("DeepRacer", options.logLevel);

    // Initialize the Bedrock service
    this.bedrockService = new BedrockService();

    // Set model ID from options, env, or default
    this.modelId =
      options.modelId ||
      process.env.INFERENCE_PROFILE_ARN ||
      process.env.DEFAULT_MODEL_ID ||
      "anthropic.claude-3-sonnet-20240229-v1:0";

    // Set system prompt if provided
    if (options.systemPrompt) {
      this.bedrockService.setSystemPrompt(options.systemPrompt);
    } else {
      this.bedrockService.setSystemPrompt(
        `You are an AI driver assistant acting like a Rally navigator for an AWS DeepRacer 1/18th scale car. ` +
          `Your job is to analyze track images from the car's perspective and suggest optimal actions. ` +
          `You should consider the track features, curves, and obstacles to make driving decisions. ` +
          `Always provide output in JSON format with "speed" (1-4 m/s) and "steering_angle" (-20 to +20 degrees) as floats. ` +
          `Negative steering angles turn the car left, positive turn it right. ` +
          `Do not add + before any positive steering angle. ` +
          `Include short "reasoning" in your response to explain your decision using Rally navigator terminology.`
      );
    }

    this.logger.info(
      `🚗 DeepRacer LLM Agent initialized with model: ${this.modelId}`
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
      prompt += " Maintain context from previous images.";
    } else {
      prompt += " This is the first image. We are in a slight left turn.";
    }

    try {
      const response = await this.bedrockService.processImageSync(
        imageBuffer,
        this.modelId,
        prompt,
        true // maintain context
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
   * Get the current token usage statistics
   * @returns Object containing token usage data
   */
  public getTokenUsage(): {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  } {
    return {
      promptTokens: this.totalPromptTokens,
      completionTokens: this.totalCompletionTokens,
      totalTokens: this.totalPromptTokens + this.totalCompletionTokens,
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
   */
  reset(resetTokens: boolean = false): void {
    this.bedrockService.clearConversation();
    this.imageCount = 0;

    if (resetTokens) {
      this.totalPromptTokens = 0;
      this.totalCompletionTokens = 0;
      this.logger.info("🔄 DeepRacer agent reset (including token counts)");
    } else {
      this.logger.info("🔄 DeepRacer agent reset");
    }
  }
}

// Example usage
async function main() {
  const mainLogger = getLogger("Main");

  // Create the DeepRacer agent with a specified log level
  const logLevel = process.env.LOG_LEVEL
    ? LogLevel[process.env.LOG_LEVEL.toUpperCase() as keyof typeof LogLevel] ||
      LogLevel.INFO
    : LogLevel.INFO;
  const agent = new DeepRacerAgent({ logLevel });

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

    // Process each image in sequence
    for (let i = 0; i < 5; i++) {
      const imagePath = path.join(testImagesDir, imageFiles[i]);
      mainLogger.info(
        `\n[${i + 1}/${imageFiles.length}] 🏎️ Processing image: ${
          imageFiles[i]
        }`
      );

      const action = await agent.processImageFile(imagePath);
      mainLogger.info("Recommended action:", JSON.stringify(action, null, 2));

      // Optional: Add a small delay between processing to avoid rate limits
      if (i < imageFiles.length - 1) {
        mainLogger.debug("Waiting before processing next image...");
        await new Promise((resolve) => setTimeout(resolve, 500));
      }
    }

    // Log the total token usage
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

    // Estimate cost (example rate for illustration)
    const promptRate = 0.002 / 1000.0; // $0.002 per 1000 tokens
    const completionRate = 0.006 / 1000.0; // $0.003 per 1000 tokens
    const estimatedCost =
      tokenUsage.promptTokens * promptRate +
      tokenUsage.completionTokens * completionRate;
    mainLogger.info(`   Estimated cost:    $${estimatedCost.toFixed(4)}`);

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
