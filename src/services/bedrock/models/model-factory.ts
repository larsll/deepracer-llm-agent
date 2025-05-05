import { IModelHandler } from "../types/bedrock-types";
import { ClaudeModelHandler } from "./claude-model";
import { MistralModelHandler } from "./mistral-model";
import { NovaModelHandler } from "./nova-model";
import { Logger, getLogger } from "../../../utils/logger";

export class ModelFactory {
  private static logger: Logger = getLogger("ModelFactory");

  /**
   * Create the appropriate model handler based on the model ID
   * @param modelId The Bedrock model ID
   * @returns An appropriate model handler
   */
  static createModelHandler(modelId: string): IModelHandler {
    if (modelId.includes("claude")) {
      return new ClaudeModelHandler(modelId);
    } else if (modelId.includes("mistral")) {
      return new MistralModelHandler();
    } else if (modelId.includes("amazon.nova")) {
      return new NovaModelHandler();
    } else {
      this.logger.warn(`No specific handler for model ${modelId}, using default handler`);
      // Return a default handler, or throw an error if you want to require explicit handlers
      return new ClaudeModelHandler(modelId);
    }
  }
}