import { BaseBedrockService } from "./base-bedrock-service";
import { ModelFactory } from "./models/model-factory";
import { LogLevel } from "../../utils/logger";
import * as fs from "fs";
import { ModelMetadata } from "../../utils/model-metadata";
import { DrivingAction } from "./types/bedrock-types";

class BedrockService extends BaseBedrockService {
  private currentModelId: string = "";

  constructor(
    options: {
      metadata?: ModelMetadata;
      logLevel?: LogLevel;
    } = {}
  ) {
    super(options.logLevel);

    // Initialize with metadata if provided
    if (options.metadata) {
      this.initializeWithMetadata(options.metadata);
    }
  }

  /**
   * Initialize the service with model metadata
   */
  initializeWithMetadata(metadata: ModelMetadata): void {
    if (!metadata.llm_config) {
      throw new Error("LLM configuration missing in metadata");
    }

    const llmConfig = metadata.llm_config;

    // Set model ID from LLM config, with fallback to environment variables
    const modelId =
      llmConfig.model_id ||
      process.env.INFERENCE_PROFILE_ARN ||
      process.env.DEFAULT_MODEL_ID ||
      "";

    if (!modelId) {
      throw new Error("No model ID specified in metadata or environment");
    }

    this.logger.info(`Initializing BedrockService with model: ${modelId}`);
    this.currentModelId = modelId;

    // Create appropriate model handler
    const modelHandler = ModelFactory.createModelHandler(modelId);
    this.setModelHandler(modelHandler);

    // Set maximum context messages
    if (llmConfig.context_window && llmConfig.context_window > 0) {
      this.setMaxContextMessages(llmConfig.context_window);
    }

    // Set system prompt
    if (Array.isArray(llmConfig.system_prompt)) {
      this.setSystemPrompt(llmConfig.system_prompt.join("\n"));
    } else if (llmConfig.system_prompt) {
      this.setSystemPrompt(llmConfig.system_prompt);
    }

    // Set action space and type
    if (metadata.action_space && metadata.action_space_type) {
      this.setActionSpace(metadata.action_space, metadata.action_space_type);
    } else {
      throw new Error("No metadata found for action space or type");
    }

  }

  /**
   * Process an image directly with Bedrock using base64 encoding
   * @param imageBuffer The image as a buffer
   * @param prompt Instructions for the model
   */
  async processImageSync(
    imageBuffer: Buffer,
    prompt: string = "Analyze this image"
  ): Promise<DrivingAction> {
    if (!this.modelHandler) {
      throw new Error("Model handler not set. Initialize with metadata first.");
    }

    // Process using the base class method
    const response = await super.processImageSync(
      imageBuffer,
      this.currentModelId,
      prompt
    );
    return response;
  }
}

export default BedrockService;
