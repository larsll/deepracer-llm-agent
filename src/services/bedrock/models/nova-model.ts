import { Logger, getLogger } from "../../../utils/logger";
import { ActionSpace, ActionSpaceType } from "../../../utils/model-metadata";
import {
  IModelHandler,
  Message,
  DrivingAction,
  TokenUsageData,
} from "../types/bedrock-types";

export class NovaModelHandler implements IModelHandler {
  private systemPrompt: string = "You are an AI driver assistant.";
  private maxContextMessages: number = 0;
  private conversationContext: Message[] = [];
  private logger: Logger;
  private actionSpace?: ActionSpace;
  private actionSpaceType?: ActionSpaceType;

  constructor() {
    this.logger = getLogger("Nova");
    this.logger.debug("Initialized Nova model handler");
  }

  getModelType(): string {
    return "nova";
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
    };
  }

  createPayload(userMessage: Message): any {
    // Amazon Nova models payload
    return {
      inferenceConfig: {
        max_new_tokens: parseInt(process.env.MAX_TOKENS || "1000"),
      },
      messages: [
        {
          role: "user",
          content: [
            {
              text: this.systemPrompt,
            },
            {
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
    };
  }

  processResponse(response: any, userMessage: Message): any {
    // Add to conversation context if maintaining context
    if (this.maxContextMessages > 0) {
      // Add assistant message
      const assistantMessage: Message = {
        role: "assistant",
        content: [
          {
            text:
              response.output?.message?.content?.[0]?.text ||
              JSON.stringify(response),
          },
        ],
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
    if (response.output?.message?.content) {
      const content = response.output.message.content[0]?.text || "";
      this.logger.debug(
        "Raw content from Nova model:",
        content.substring(0, 200)
      );

      // Try to extract JSON from content - either from code blocks or directly
      try {
        // First try to find JSON in code blocks
        const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)\s*```|(\{[\s\S]*?\})/);
        
        if (jsonMatch) {
          // Parse JSON from code block
          const jsonString = (jsonMatch[1] || jsonMatch[2]).trim();
          this.logger.debug("Extracted JSON from formatted block");
          return JSON.parse(jsonString);
        } 
        
        // If no code block found, try parsing the entire content
        this.logger.debug("Attempting to parse entire content as JSON");
        return JSON.parse(content.trim());
        
      } catch (error) {
        this.logger.error("Failed to parse Nova response as JSON:", error);
        this.logger.debug("Raw content:", content);
        throw new Error("No valid JSON found in Nova response");
      }
    } else {
      this.logger.error(
        "Unexpected Nova response structure:",
        JSON.stringify(response).substring(0, 200)
      );
      throw new Error("Unexpected Nova response structure");
    }
  }

  /**
   * Extract token usage from Nova response
   */
  extractTokenUsage(response: any): TokenUsageData {
    const result: TokenUsageData = {
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
    };

    // Handle Amazon Nova format
    if (response.usage?.inputTokens && response.usage?.outputTokens) {
      result.promptTokens = response.usage.inputTokens;
      result.completionTokens = response.usage.outputTokens;
      result.totalTokens = result.promptTokens + result.completionTokens;
      return result;
    }

    this.logger.debug("Could not determine token usage from Nova response");
    return result;
  }
}
