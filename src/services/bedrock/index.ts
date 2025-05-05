import { BaseBedrockService } from "./base-bedrock-service";
import { ModelFactory } from "./models/model-factory";
import { IModelHandler } from "./types/bedrock-types";
import { LogLevel } from "../../utils/logger";
import * as fs from "fs";

class BedrockService extends BaseBedrockService {
  private currentModelId: string = "";

  constructor(logLevel?: LogLevel) {
    super(logLevel);
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
    // If modelId changed or modelHandler not set, update it
    if (modelId !== this.currentModelId || !this.modelHandler) {
      this.logger.info(`Setting model handler for ${modelId}`);
      this.currentModelId = modelId;
      
      // Create appropriate model handler
      const modelHandler = ModelFactory.createModelHandler(modelId);
      
      // Transfer settings to the new handler
      if (this.systemPrompt) {
        modelHandler.setSystemPrompt(this.systemPrompt);
      }
      
      if (this.maxContextMessages > 0) {
        modelHandler.setMaxContextMessages(this.maxContextMessages);
      }
      
      // Set the model handler
      this.setModelHandler(modelHandler);
    }
    
    // Process using the base class method
    const response = await super.processImageSync(imageBuffer, modelId, prompt);
    return response;
  }
  
  /**
   * Extract driving action from response using the current model handler
   * @param response The raw model response
   * @returns A driving action object
   */
  extractDrivingAction(response: any): any {
    if (!this.modelHandler) {
      throw new Error("Model handler not set");
    }
    
    try {
      return this.modelHandler.extractDrivingAction(response);
    } catch (error) {
      this.logger.error("Failed to extract driving action:", error);
      
      // Last resort attempt
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
      
      throw new Error("Failed to extract driving action");
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
}

export default BedrockService;