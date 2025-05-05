import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from "@aws-sdk/client-bedrock-runtime";
import * as fs from "fs";
import { Logger, LogLevel, getLogger } from "../utils/logger";
const crypto = require("crypto");

class BedrockService {
  private bedrockClient: BedrockRuntimeClient;
  private conversationContext: Array<any> = [];
  private systemPrompt: string | null = null;
  private logger: Logger;
  private maxContextMessages: number = 0;

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
   * Set a system prompt to be used for all subsequent interactions
   * @param systemPrompt The system prompt text
   */
  setSystemPrompt(systemPrompt: string): void {
    this.systemPrompt = systemPrompt;
    this.logger.info("System prompt set");
  }

  setMaxContextMessages(maxMessages: number): void {
    this.logger.info(`Max context messages set to ${maxMessages}`);
    this.maxContextMessages = maxMessages;
  }

  /**
   * Clear the conversation history
   */
  clearConversation(): void {
    this.conversationContext = [];
    this.logger.info("Conversation context cleared");
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
    // Convert image to base64
    const base64Image = imageBuffer.toString("base64");
    const md5Hash = crypto.createHash("md5").update(base64Image).digest("hex");
    this.logger.debug("Image MD5 hash: ", md5Hash);
    // Create request payload based on the model
    const payload = this.createPayloadForModel(modelId, base64Image, prompt);
    //this.logger.debug("Payload created for model:", payload);

    // Invoke Bedrock model with timeout
    const command = new InvokeModelCommand({
      modelId,
      contentType: "application/json",
      accept: "application/json",
      body: JSON.stringify(payload),
    });

    try {
      this.logger.debug(`Processing image with model: ${modelId}`);
      const timeout = parseInt(process.env.TIMEOUT_MS || "30000");

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
      this.logger.debug("Response received from Bedrock:", responseBody);

      // Add to conversation context if maintaining context
      if (this.maxContextMessages > 0) {
        // Add user message to context - use exact same format as in the request
        let userMessage;

        if (modelId.includes("amazon.nova")) {
          // Format specifically for Nova
          userMessage = {
            role: "user",
            content: [
              {
                image: {
                  format: "jpeg",
                  source: {
                    bytes: base64Image,
                  },
                },
              },
              {
                text: prompt,
              },
            ],
          };
        } else {
          // Format for other models
          userMessage = {
            role: "user",
            content: [
              { type: "text", text: prompt },
              modelId.includes("claude")
                ? {
                    type: "image",
                    source: {
                      type: "base64",
                      media_type: "image/jpeg",
                      data: base64Image,
                    },
                  }
                : {
                    type: "image_url",
                    image_url: {
                      url: `data:image/jpeg;base64,${base64Image}`,
                    },
                  },
            ],
          };
        }

        // Add assistant response to context
        let assistantMessage;
        if (modelId.includes("claude-3-7")) {
          // For Claude 3.7, we need to handle the conversation context differently
          // since it doesn't use system message in the messages array
          assistantMessage = {
            role: "assistant",
            content:
              responseBody.content?.[0]?.text || JSON.stringify(responseBody),
          };
        } else if (modelId.includes("claude")) {
          assistantMessage = {
            role: "assistant",
            content:
              responseBody.content?.[0]?.text || JSON.stringify(responseBody),
          };
        } else if (modelId.includes("mistral")) {
          assistantMessage = {
            role: "assistant",
            content:
              responseBody.choices?.[0]?.message?.content ||
              responseBody.messages?.[0]?.content ||
              JSON.stringify(responseBody),
          };
        } else if (modelId.includes("amazon.nova")) {
          assistantMessage = {
            role: "assistant",
            content: [
              {
                text:
                  responseBody.output?.message?.content?.[0]?.text ||
                  JSON.stringify(responseBody),
              },
            ],
          };
        }

        // Replace entire context with just this exchange when maintaining context
        // This prevents accumulating too many messages
        this.conversationContext.push(userMessage, assistantMessage);

        this.logger.debug(
          `Added to conversation context (${this.conversationContext.length} messages)`
        );
      }

      return responseBody;
    } catch (error) {
      this.logger.error("Error processing image with Bedrock:", error);
      throw error;
    }
  }

  /**
   * Process an image from a file path
   */
  async processImageFromFile(
    filePath: string,
    modelId: string,
    prompt: string = "Analyze this image"
  ): Promise<any> {
    const imageBuffer = fs.readFileSync(filePath);
    return this.processImageSync(imageBuffer, modelId, prompt);
  }

  /**
   * Process a sequence of images with context
   * @param imagePaths Array of paths to image files
   * @param modelId The Bedrock model ID to use
   * @param prompts Array of prompts for each image, or a single prompt to use for all
   */
  async processImageSequence(
    imagePaths: string[],
    modelId: string,
    prompts: string | string[]
  ): Promise<any[]> {
    // Clear previous context
    this.clearConversation();

    const results = [];

    for (let i = 0; i < imagePaths.length; i++) {
      const imagePath = imagePaths[i];

      // Use either the specific prompt for this image or the default prompt
      const prompt = Array.isArray(prompts) ? prompts[i] : prompts;

      this.logger.info(
        `Processing image ${i + 1}/${imagePaths.length}: ${imagePath}`
      );
      const result = await this.processImageFromFile(
        imagePath,
        modelId,
        prompt
      );
      results.push(result);
    }

    return results;
  }

  /**
   * Create the appropriate payload structure based on the model
   */
  private createPayloadForModel(
    modelId: string,
    base64Image: string,
    prompt: string
  ): any {
    // Store only the most recent exchange in this format rather than cumulative history
    if (modelId.includes("claude-3-7")) {
      // Claude 3.7 models require system prompt as a top-level parameter
      return {
        anthropic_version: "bedrock-2023-05-31",
        system: this.systemPrompt || "You are an AI driver assistant.",
        messages: [
          ...(this.conversationContext.length > 0
            ? this.conversationContext.slice(-this.maxContextMessages * 2)
            : []),
          {
            role: "user",
            content: [
              { type: "text", text: prompt },
              {
                type: "image",
                source: {
                  type: "base64",
                  media_type: "image/jpeg",
                  data: base64Image,
                },
              },
            ],
          },
        ],
        max_tokens: parseInt(process.env.MAX_TOKENS || "1000"),
      };
    } else if (modelId.includes("mistral.pixtral")) {
      // Mistral Pixtral models payload
      return {
        messages: [
          {
            role: "system",
            content: this.systemPrompt || "You are an AI driver assistant.",
          },
          ...(this.conversationContext.length > 0
            ? this.conversationContext.slice(-this.maxContextMessages * 2)
            : []),
          {
            role: "user",
            content: [
              { type: "text", text: prompt },
              {
                type: "image_url",
                image_url: {
                  url: `data:image/jpeg;base64,${base64Image}`,
                },
              },
            ],
          },
        ],
      };
    } else if (modelId.includes("amazon.nova")) {
      // Amazon Nova models payload - Updated to match Nova Lite's expected format
      return {
        inferenceConfig: {
          max_new_tokens: parseInt(process.env.MAX_TOKENS || "1000"),
        },
        messages: [
          {
            role: "user",
            content: [
              {
                text: this.systemPrompt || "You are an AI driver assistant.",
              },
            ],
          },
          ...(this.conversationContext.length > 0
            ? this.conversationContext.slice(-this.maxContextMessages * 2)
            : []),
          {
            role: "user",
            content: [
              {
                text: prompt,
              },
              {
                image: {
                  format: "jpeg",
                  source: {
                    bytes: base64Image,
                  },
                },
              },
            ],
          },
        ],
      };
    } else {
      // Default payload structure
      return {
        prompt: `${
          this.conversationContext.length > 0
            ? "Previous analysis: " +
              JSON.stringify(this.conversationContext.slice(-1)[0]) +
              "\n"
            : ""
        }${prompt}`,
        image_data: base64Image,
      };
    }
  }
}

export default BedrockService;
