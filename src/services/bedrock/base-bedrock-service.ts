import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from "@aws-sdk/client-bedrock-runtime";
import { Logger, LogLevel, getLogger } from "../../utils/logger";
import { IModelHandler, TokenUsageData } from "./types/bedrock-types";

/**
 * Base Bedrock service with core functionality
 */
export abstract class BaseBedrockService {
  protected bedrockClient: BedrockRuntimeClient;
  protected conversationContext: Array<any> = [];
  protected systemPrompt: string | null = null;
  protected logger: Logger;
  protected maxContextMessages: number = 0;
  protected modelHandler: IModelHandler | null = null;

  // Track token usage
  private promptTokens: number = 0;
  private completionTokens: number = 0;
  private totalTokens: number = 0;

  constructor(logLevel?: LogLevel) {
    // Initialize logger
    this.logger = getLogger("Bedrock", logLevel);

    // Use credentials from .env file (loaded by dotenv)
    this.bedrockClient = new BedrockRuntimeClient({
      region: process.env.AWS_REGION,
      credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID as string,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY as string,
      },
    });

    this.logger.info(
      `Initialized Bedrock client for region: ${process.env.AWS_REGION}`
    );
  }

  /**
   * Sets the appropriate model handler based on modelId
   * @param modelHandler The ID of the model to use
   */
  setModelHandler(modelHandler: IModelHandler): void {
    this.modelHandler = modelHandler;
    this.logger.info(`Set model handler for: ${modelHandler.getModelType()}`);
  }

  /**
   * Set a system prompt to be used for all subsequent interactions
   * @param systemPrompt The system prompt text
   */
  setSystemPrompt(systemPrompt: string): void {
    this.systemPrompt = systemPrompt;
    if (this.modelHandler) {
      this.modelHandler.setSystemPrompt(systemPrompt);
    }
    this.logger.info("System prompt set");
  }

  /**
   * Set maximum number of context messages to retain
   * @param maxMessages The maximum number of messages to keep in context
   */
  setMaxContextMessages(maxMessages: number): void {
    this.logger.info(`Max context messages set to ${maxMessages}`);
    this.maxContextMessages = maxMessages;
    if (this.modelHandler) {
      this.modelHandler.setMaxContextMessages(maxMessages);
    }
  }

  /**
   * Clear the conversation history
   */
  clearConversation(): void {
    this.conversationContext = [];
    if (this.modelHandler) {
      this.modelHandler.clearConversation();
    }
    this.logger.info("Conversation context cleared");
  }

  /**
   * Track token usage from a response using the model handler
   */
  trackTokenUsage(response: any): void {
    if (!this.modelHandler) return;

    try {
      const usage = this.modelHandler.extractTokenUsage(response);

      this.promptTokens += usage.promptTokens;
      this.completionTokens += usage.completionTokens;
      this.totalTokens += usage.totalTokens;

      this.logger.debug(
        `Token usage for this request - Prompt: ${usage.promptTokens}, Completion: ${usage.completionTokens}`
      );
    } catch (error) {
      this.logger.warn(`Failed to track token usage: ${error}`);
    }
  }

  /**
   * Get current token usage statistics
   */
  getTokenUsage(): TokenUsageData {
    return {
      promptTokens: this.promptTokens,
      completionTokens: this.completionTokens,
      totalTokens: this.totalTokens,
    };
  }

  /**
   * Reset token usage counters
   */
  resetTokenUsage(): void {
    this.promptTokens = 0;
    this.completionTokens = 0;
    this.totalTokens = 0;
  }

  /**
   * Process an image directly with Bedrock using base64 encoding
   * @param imageBuffer The image as a buffer
   * @param modelId The Bedrock model ID to use
   * @param prompt Instructions for the model
   */
  async processImageSync(
    imageBuffer: Buffer,
    modelId: string,
    prompt: string = "Analyze this image"
  ): Promise<any> {
    if (!this.modelHandler) {
      throw new Error("Model handler not set. Call setModelHandler first.");
    }

    // Convert image to base64
    const base64Image = imageBuffer.toString("base64");

    try {
      this.logger.debug(`Processing image with model: ${modelId}`);
      const timeout = parseInt(process.env.TIMEOUT_MS || "30000");

      // Create user message
      const userMessage = this.modelHandler.createUserMessage(prompt, base64Image);

      // Create payload based on model type - now using userMessage
      const payload = this.modelHandler.createPayload(userMessage);

      // Create command for Bedrock
      const command = new InvokeModelCommand({
        modelId,
        contentType: "application/json",
        accept: "application/json",
        body: JSON.stringify(payload),
      });

      // Create a promise with timeout
      const timeoutPromise = new Promise((_, reject) =>
        setTimeout(() => reject(new Error("Request timed out")), timeout)
      );

      // Race between the actual request and the timeout
      const response = (await Promise.race([
        this.bedrockClient.send(command),
        timeoutPromise,
      ])) as any;

      // Parse the response body
      const responseBody = JSON.parse(new TextDecoder().decode(response.body));
      this.logger.debug("Response received from Bedrock");

      // Process the response with the model handler - now using userMessage
      const parsedResponse = this.modelHandler.processResponse(
        responseBody,
        userMessage
      );

      // Track token usage
      this.trackTokenUsage(responseBody);

      return parsedResponse;
    } catch (error) {
      this.logger.error("Error processing image with Bedrock:", error);
      throw error;
    }
  }
}
