import { Logger, getLogger } from "../../../utils/logger";
import { ActionSpace, ActionSpaceType } from "../../../utils/model-metadata";
import {
  IModelHandler,
  Message,
  DrivingAction,
  TokenUsageData,
} from "../types/bedrock-types";
import { extractJsonFromLlmResponse } from "../../../utils/json-extractor";

export class MistralModelHandler implements IModelHandler {
  private systemPrompt: string = "You are an AI driver assistant.";
  private maxContextMessages: number = 0;
  private conversationContext: Message[] = [];
  private logger: Logger;
  private actionSpace?: ActionSpace;
  private actionSpaceType?: ActionSpaceType;

  constructor() {
    this.logger = getLogger("Mistral");
    this.logger.debug("Initialized Mistral model handler");
  }

  getModelType(): string {
    return "mistral";
  }

  setSystemPrompt(prompt: string): void {
    this.systemPrompt = prompt;
  }

  setMaxContextMessages(max: number): void {
    this.maxContextMessages = max;
  }

  setActionSpace(actionSpace: ActionSpace): void {
    this.actionSpace = actionSpace;
  }

  setActionSpaceType(actionSpaceType: ActionSpaceType): void {
    this.actionSpaceType = actionSpaceType;
  }

  clearConversation(): void {
    this.conversationContext = [];
  }

  createUserMessage(prompt: string, base64Image: string): Message {
    return {
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
    };
  }

  createPayload(userMessage: Message): any {
    // Mistral Pixtral models payload
    return {
      messages: [
        {
          role: "system",
          content: this.systemPrompt,
        },
        {
          role: "user",
          content: [
            {
              type: "text",
              text: JSON.stringify({
                action_space_type: this.actionSpaceType,
                action_space: this.actionSpace,
              }),
            },
          ],
        },
        ...(this.conversationContext.length > 0 && this.maxContextMessages > 0
          ? this.conversationContext.slice(-this.maxContextMessages)
          : []),
        userMessage,
      ],
      max_tokens: parseInt(process.env.MAX_TOKENS || "1000"),
    };
  }

  processResponse(response: any, userMessage: Message): any {
    // Add to conversation context if maintaining context
    if (this.maxContextMessages > 0) {
      // Add assistant message
      const assistantMessage: Message = {
        role: "assistant",
        content:
          response.choices?.[0]?.message?.content || JSON.stringify(response),
      };

      this.conversationContext.push(userMessage, assistantMessage);

      // Limit context length if needed
      if (this.maxContextMessages > 0) {
        this.conversationContext = this.conversationContext.slice(
          -this.maxContextMessages
        );
      }
    }

    return response;
  }

  extractDrivingAction(response: any): DrivingAction {
    if (response.choices && response.choices[0]?.message?.content) {
      const content = response.choices[0].message.content;
      return extractJsonFromLlmResponse<DrivingAction>(
        content,
        this.logger,
        "Mistral"
      );
    } else {
      this.logger.error(
        "Unexpected Mistral response structure:",
        JSON.stringify(response).substring(0, 200)
      );
      throw new Error("Unexpected Mistral response structure");
    }
  }

  /**
   * Extract token usage from Mistral response
   */
  extractTokenUsage(response: any): TokenUsageData {
    const result: TokenUsageData = {
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
    };

    // Handle standard format
    if (response.usage?.prompt_tokens && response.usage?.completion_tokens) {
      result.promptTokens = response.usage.prompt_tokens;
      result.completionTokens = response.usage.completion_tokens;
      result.totalTokens = result.promptTokens + result.completionTokens;
      return result;
    }

    // Handle Mistral API format with only total tokens
    if (response.usage?.total_tokens) {
      // For Mistral, we might only know the total tokens
      // Make an estimate of the split between prompt and completion
      const totalTokens = response.usage.total_tokens;
      // Default to assuming 2/3 are prompt tokens if we don't know better
      result.promptTokens =
        response.usage.prompt_tokens || Math.floor(totalTokens * 0.67);
      result.completionTokens =
        response.usage.completion_tokens || totalTokens - result.promptTokens;
      result.totalTokens = totalTokens;
      return result;
    }

    this.logger.debug("Could not determine token usage from Mistral response");
    return result;
  }
}
