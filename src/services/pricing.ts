import { PricingClient, GetProductsCommand } from "@aws-sdk/client-pricing";
import { Logger, getLogger } from "../utils/logger";
import awsConfig from "../utils/aws-config";
import { AwsCredentialIdentity, AwsCredentialIdentityProvider } from "@aws-sdk/types";

// Define interface for token pricing
export interface TokenPricing {
  promptRate: number; // Cost per 1000 prompt tokens
  completionRate: number; // Cost per 1000 completion tokens
}

/**
 * Service to fetch and manage pricing information for LLM tokens
 */
export class PricingService {
  private pricingClient: PricingClient;
  private logger: Logger;

  // Default pricing rates if API calls fail
  private defaultPricing: TokenPricing = {
    promptRate: 0.002, // Default fallback rate per 1000 tokens
    completionRate: 0.006, // Default fallback rate per 1000 tokens
  };

  // Current pricing for the model
  private currentPricing: TokenPricing;

  constructor(region?: string) {
    this.logger = getLogger("PricingService");

    // Hard-coded as pricing is only in us-east-1
    const clientRegion = "us-east-1";
    
    // Initialize pricing client with the configured region and credentials
    this.pricingClient = new PricingClient({
      region: clientRegion,
      credentials: awsConfig.credentials as AwsCredentialIdentity | AwsCredentialIdentityProvider
    });

    this.logger.debug(`Initialized pricing client with region: ${clientRegion}`);

    // Initialize with default pricing
    this.currentPricing = { ...this.defaultPricing };
  }

  /**
   * Get the current token pricing
   * @returns The current token pricing information
   */
  public getPricing(): TokenPricing {
    return { ...this.currentPricing };
  }

  /**
   * Get the model name from the model ID for pricing lookup
   * @param modelId The model identifier
   * @returns The standardized model name
   */
  private getModelName(modelId: string): string {
    const modelMapping: { [key: string]: string } = {
      "amazon.nova-lite": "Nova Lite",
      "amazon.nova-pro": "Nova Pro",
      "anthropic.claude-3-7-sonnet": "Claude 3 Sonnet",
      "mistral.pixtral-large": "Pixtral Large 25.02",
    };

    // Extract model name from ARN if applicable
    let modelName = "Unknown";
    if (modelId.includes("arn:aws:bedrock")) {
      modelName = modelId.split("/").pop()?.split(":")[0] || "";
      this.logger.debug(`Extracted model name from ARN: ${modelName}`);
    } else {
      modelName = modelId;
    }

    // Match model ID directly
    for (const key in modelMapping) {
      if (modelName.includes(key)) {
        return modelMapping[key];
      }
    }

    return "Unknown"; // Default fallback
  }

  /**
   * Reset pricing to defaults
   */
  public resetToDefaults(): void {
    this.currentPricing = { ...this.defaultPricing };
    this.logger.debug("Pricing reset to defaults");
  }

  /**
   * Fetch pricing information from AWS Price List API for the specified model
   * @param modelId The model identifier
   * @param region The AWS region to fetch pricing for (default: 'eu-central-1')
   * @returns Promise resolving to token pricing information
   */
  public async loadModelPricing(
    modelId: string,
    region: string = "eu-central-1"
  ): Promise<TokenPricing> {
    try {
      this.logger.debug(
        `Fetching pricing data for model: ${modelId} in region ${region}`
      );

      // Map model ID to service code
      const serviceCode = "AmazonBedrock";

      // Get model family from model ID (now handles ARN format)
      const model = this.getModelName(modelId);

      const pricingRequest = {
        ServiceCode: serviceCode,
        Filters: [
          {
            Type: "TERM_MATCH" as const,
            Field: "model",
            Value: model,
          },
          {
            Type: "TERM_MATCH" as const,
            Field: "regionCode",
            Value: region,
          },
        ],
      };

      this.logger.debug(`Using model name for pricing lookup: ${model}`);

      const command = new GetProductsCommand(pricingRequest);

      try {
        const response = await this.pricingClient.send(command);

        if (response.PriceList && response.PriceList.length > 0) {
          let foundPromptPrice = false;
          let foundCompletionPrice = false;

          this.logger.debug(
            `Found ${response.PriceList.length} pricing items to parse`
          );

          // Use default pricing as starting point (in case we only find one of the rates)
          const newPricing: TokenPricing = { ...this.defaultPricing };

          // Parse the pricing data
          for (const priceItem of response.PriceList) {
            const priceData = JSON.parse(priceItem);

            // Check if this is input or output token pricing
            const usageType = priceData.product?.attributes?.usagetype || "";
            const inferenceType =
              priceData.product?.attributes?.inferenceType || "";
            const feature = priceData.product?.attributes?.feature || "";

            // Skip batch inference pricing if we're doing on-demand inference
            if (feature.includes("Batch") && !modelId.includes("batch")) {
              this.logger.debug(`Skipping batch pricing: ${usageType}`);
              continue;
            }

            // Skip cache read pricing
            if (
              inferenceType.includes("cache") ||
              usageType.includes("cache")
            ) {
              this.logger.debug(`Skipping cache pricing: ${usageType}`);
              continue;
            }

            // Extract pricing information from the price dimensions
            if (priceData?.terms?.OnDemand) {
              const onDemandKey = Object.keys(priceData.terms.OnDemand)[0];
              if (!onDemandKey) continue;

              const priceDimensions =
                priceData.terms.OnDemand[onDemandKey].priceDimensions;
              if (!priceDimensions) continue;

              const priceDimensionKey = Object.keys(priceDimensions)[0];
              if (!priceDimensionKey) continue;

              const priceDimension = priceDimensions[priceDimensionKey];

              if (
                !priceDimension ||
                !priceDimension.pricePerUnit ||
                !priceDimension.pricePerUnit.USD
              ) {
                this.logger.debug("Invalid price dimension structure");
                continue;
              }

              const pricePerUnit = parseFloat(priceDimension.pricePerUnit.USD);

              // Determine if this is input or output token pricing
              if (
                (inferenceType.includes("Input") ||
                  usageType.includes("input")) &&
                !inferenceType.includes("cache") &&
                !usageType.includes("cache")
              ) {
                newPricing.promptRate = pricePerUnit;
                this.logger.debug(
                  `Found input token price: $${pricePerUnit}/1K tokens (${usageType})`
                );
                foundPromptPrice = true;
              } else if (
                (inferenceType.includes("Output") ||
                  usageType.includes("output")) &&
                !inferenceType.includes("cache") &&
                !usageType.includes("cache")
              ) {
                newPricing.completionRate = pricePerUnit;
                this.logger.debug(
                  `Found output token price: $${pricePerUnit}/1K tokens (${usageType})`
                );
                foundCompletionPrice = true;
              }
            }
          }

          // Update the current pricing
          this.currentPricing = newPricing;

          if (foundPromptPrice && foundCompletionPrice) {
            this.logger.info(
              `Loaded pricing data: Input tokens $${this.currentPricing.promptRate}/1K tokens, Output tokens $${this.currentPricing.completionRate}/1K tokens`
            );
          } else if (foundPromptPrice) {
            this.logger.warn(
              `Only found input token pricing. Using default for output tokens.`
            );
          } else if (foundCompletionPrice) {
            this.logger.warn(
              `Only found output token pricing. Using default for input tokens.`
            );
          } else {
            this.logger.warn(
              `No applicable pricing data found for model ${modelId}, using defaults`
            );
          }
        } else {
          this.logger.warn(
            `No pricing data found for model ${modelId}, using defaults`
          );
        }
      } catch (error) {
        this.logger.warn(
          `Error fetching pricing data: ${error}. Using default pricing.`
        );
      }

      return this.currentPricing;
    } catch (error) {
      this.logger.warn(
        `Failed to load model pricing: ${error}. Using default pricing.`
      );
      return this.defaultPricing;
    }
  }

  /**
   * Calculate the cost of token usage
   * @param promptTokens Number of prompt tokens used
   * @param completionTokens Number of completion tokens used
   * @returns Cost information including breakdown and total
   */
  public calculateCost(
    promptTokens: number,
    completionTokens: number
  ): {
    promptCost: number;
    completionCost: number;
    totalCost: number;
  } {
    const promptCost = promptTokens * (this.currentPricing.promptRate / 1000);
    const completionCost =
      completionTokens * (this.currentPricing.completionRate / 1000);

    return {
      promptCost,
      completionCost,
      totalCost: promptCost + completionCost,
    };
  }
}

export default PricingService;
