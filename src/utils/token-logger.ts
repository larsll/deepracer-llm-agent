import { Logger } from "./logger";

export class TokenLogger {
  private totalPromptTokens: number = 0;
  private totalCompletionTokens: number = 0;
  private logger: Logger;

  constructor(logger: Logger) {
    this.logger = logger;
  }

  trackTokenUsage(response: any): void {
    if (!response) return;

    try {
      if (response.usage) {
        const promptTokens = response.usage.inputTokens || response.usage.prompt_tokens || 0;
        const completionTokens = response.usage.outputTokens || response.usage.completion_tokens || 0;

        this.totalPromptTokens += promptTokens;
        this.totalCompletionTokens += completionTokens;

        this.logger.debug(
          `Tokens - Prompt: ${promptTokens}, Completion: ${completionTokens}, Total: ${
            promptTokens + completionTokens
          }`
        );
      } else if (response.usage_metadata) {
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

  getTokenUsage(): {
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
}