import { Logger, getLogger } from "../../../utils/logger";
import { ActionSpace, ActionSpaceType } from "../../../utils/model-metadata";
import {
  IModelHandler,
  Message,
  DrivingAction,
  TokenUsageData,
} from "../types/bedrock-types";
import { extractJsonFromLlmResponse } from "../../../utils/json-extractor";

export class ClaudeModelHandler implements IModelHandler {
  private systemPrompt: string = "You are an AI driver assistant.";
  private maxContextMessages: number = 0;
  private conversationContext: Message[] = [];
  private logger: Logger;
  private is37Model: boolean;
  private actionSpace?: ActionSpace;
  private actionSpaceType?: ActionSpaceType;

  constructor(modelId: string) {
    this.logger = getLogger("Claude");
    this.is37Model = modelId.includes("claude-3-7");
    this.logger.debug(
      `Initialized Claude model handler (${
        this.is37Model ? "3.7" : "standard"
      } format)`
    );
  }

  getModelType(): string {
    return "claude";
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
          type: "image",
          source: {
            type: "base64",
            media_type: "image/jpeg",
            data: base64Image,
          },
        },
      ],
    };
  }

  createPayload(userMessage: Message): any {
    if (this.is37Model) {
      // Claude 3.7 models require system prompt as a top-level parameter
      return {
        anthropic_version: "bedrock-2023-05-31",
        system: this.systemPrompt,
        messages: [
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
    } else {
      // Older Claude models
      return {
        anthropic_version: "bedrock-2023-05-31",
        system: this.systemPrompt,
        messages: [
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
          ...(this.conversationContext.length > 0
            ? this.conversationContext.slice(-this.maxContextMessages)
            : []),
          userMessage,
        ],
        max_tokens: parseInt(process.env.MAX_TOKENS || "1000"),
      };
    }
  }

  processResponse(response: any, userMessage: Message): any {
    // Add to conversation context if maintaining context
    if (this.maxContextMessages > 0) {
      // Add user message to context - now using the passed userMessage

      // Add assistant response to context
      const assistantMessage: Message = {
        role: "assistant",
        content: response.content?.[0]?.text || JSON.stringify(response),
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
    if (response.content && response.content.length > 0) {
      const content = response.content[0]?.text || "";
      this.logger.debug("Extracting driving action from Claude response");
      return extractJsonFromLlmResponse<DrivingAction>(
        content,
        this.logger,
        "Claude"
      );
    } else {
      this.logger.error(
        "Unexpected Claude response structure:",
        JSON.stringify(response).substring(0, 200)
      );
      throw new Error("Unexpected Claude response structure");
    }
  }

  /**
   * Extract token usage from Claude response
   */
  extractTokenUsage(response: any): TokenUsageData {
    const result: TokenUsageData = {
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
    };

    // Handle Claude's token usage format
    if (
      response.usage &&
      (response.usage.input_tokens !== undefined ||
        response.usage.output_tokens !== undefined)
    ) {
      result.promptTokens = response.usage.input_tokens || 0;
      result.completionTokens = response.usage.output_tokens || 0;
      result.totalTokens = result.promptTokens + result.completionTokens;
      return result;
    }

    // Handle Amazon Bedrock Claude format (where tokens might be in the response metadata)
    if (
      response.metadata?.usage?.inputTokens &&
      response.metadata?.usage?.outputTokens
    ) {
      result.promptTokens = response.metadata.usage.inputTokens;
      result.completionTokens = response.metadata.usage.outputTokens;
      result.totalTokens = result.promptTokens + result.completionTokens;
      return result;
    }

    // Handle Anthropic Direct API format
    if (response.usage?.input_tokens && response.usage?.output_tokens) {
      result.promptTokens = response.usage.input_tokens;
      result.completionTokens = response.usage.output_tokens;
      result.totalTokens = result.promptTokens + result.completionTokens;
      return result;
    }

    this.logger.debug("Could not determine token usage from Claude response");
    return result;
  }
}
