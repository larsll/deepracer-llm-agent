import { Logger } from "./logger";

/**
 * Interface for token usage data
 */
export interface TokenUsageData {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

/**
 * Token logger for tracking token usage across API calls
 */
export class TokenLogger {
  private promptTokens: number = 0;
  private completionTokens: number = 0;
  private totalTokens: number = 0;
  private logger: Logger;

  constructor(logger: Logger) {
    this.logger = logger;
  }

  /**
   * Track token usage from an API response
   * @param response The API response containing token usage data
   */
  trackTokenUsage(response: any): void {
    try {
      if (!response) return;

      // Handle Claude's specific token usage format
      if (
        response.usage &&
        (response.usage.input_tokens !== undefined ||
          response.usage.output_tokens !== undefined)
      ) {
        // Extract Claude's token usage
        const promptTokens = response.usage.input_tokens || 0;
        const completionTokens = response.usage.output_tokens || 0;

        this.logger.debug(
          `Token usage for this request - Prompt: ${promptTokens}, Completion: ${completionTokens}`
        );

        // Add to running totals
        this.promptTokens += promptTokens;
        this.completionTokens += completionTokens;
        this.totalTokens += promptTokens + completionTokens;

        return;
      }

      // Handle Anthropic Direct API format
      if (response.usage?.input_tokens && response.usage?.output_tokens) {
        const promptTokens = response.usage.input_tokens;
        const completionTokens = response.usage.output_tokens;

        this.promptTokens += promptTokens;
        this.completionTokens += completionTokens;
        this.totalTokens += promptTokens + completionTokens;

        return;
      }

      // Handle standard OpenAI-like format
      if (response.usage?.prompt_tokens && response.usage?.completion_tokens) {
        const promptTokens = response.usage.prompt_tokens;
        const completionTokens = response.usage.completion_tokens;

        this.promptTokens += promptTokens;
        this.completionTokens += completionTokens;
        this.totalTokens += promptTokens + completionTokens;

        return;
      }

      // Handle PassthroughAPIModel format where tokenUsage is passed directly
      if (
        typeof response.promptTokens === "number" &&
        typeof response.completionTokens === "number"
      ) {
        this.promptTokens += response.promptTokens;
        this.completionTokens += response.completionTokens;
        this.totalTokens += response.promptTokens + response.completionTokens;

        return;
      }

      // Handle Amazon Bedrock Claude format (where tokens might be in the response metadata)
      if (
        response.metadata?.usage?.inputTokens &&
        response.metadata?.usage?.outputTokens
      ) {
        const promptTokens = response.metadata.usage.inputTokens;
        const completionTokens = response.metadata.usage.outputTokens;

        this.promptTokens += promptTokens;
        this.completionTokens += completionTokens;
        this.totalTokens += promptTokens + completionTokens;

        return;
      }

      // Handle Amazon Nova format
      if (response.usage?.inputTokens && response.usage?.outputTokens) {
        const promptTokens = response.usage.inputTokens;
        const completionTokens = response.usage.outputTokens;

        this.promptTokens += promptTokens;
        this.completionTokens += completionTokens;
        this.totalTokens += promptTokens + completionTokens;

        return;
      }

      // Handle Mistral API format
      if (response.usage?.total_tokens) {
        // For Mistral, we might only know the total tokens
        // Make an estimate of the split between prompt and completion
        // This is a rough estimate and might not be accurate
        const totalTokens = response.usage.total_tokens;
        // Default to assuming 2/3 are prompt tokens if we don't know better
        const promptTokens =
          response.usage.prompt_tokens || Math.floor(totalTokens * 0.67);
        const completionTokens =
          response.usage.completion_tokens || totalTokens - promptTokens;

        this.promptTokens += promptTokens;
        this.completionTokens += completionTokens;
        this.totalTokens += totalTokens;

        return;
      }

      // If we got here, we couldn't identify the token usage format
      this.logger.debug(
        "Could not determine token usage from response format:",
        JSON.stringify(response).substring(0, 200) + "..."
      );
    } catch (error) {
      this.logger.warn(`Failed to track token usage: ${error}`);
    }
  }

  /**
   * Get the current token usage statistics
   * @returns Object containing token usage data
   */
  getTokenUsage(): TokenUsageData {
    return {
      promptTokens: this.promptTokens,
      completionTokens: this.completionTokens,
      totalTokens: this.totalTokens,
    };
  }

  /**
   * Reset token counters to zero
   */
  reset(): void {
    this.promptTokens = 0;
    this.completionTokens = 0;
    this.totalTokens = 0;
  }
}
