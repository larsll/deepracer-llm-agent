import { Logger } from "./logger";

/**
 * Extracts and parses JSON from LLM text responses
 * Handles various formats including code blocks and direct JSON
 * 
 * @param content The text content from an LLM response
 * @param logger Logger instance for debugging
 * @param modelName Optional model name for better error messages
 * @returns Parsed JSON object
 */
export function extractJsonFromLlmResponse<T>(
  content: string,
  logger: Logger,
  modelName: string = "LLM"
): T {
  logger.debug(
    `Raw content from ${modelName} model:`,
    content.substring(0, 200)
  );

  // Try to extract JSON from content - either from code blocks or directly
  try {
    // First try to find JSON in code blocks
    const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)\s*```|(\{[\s\S]*?\})/);
    
    if (jsonMatch) {
      // Parse JSON from code block
      const jsonString = (jsonMatch[1] || jsonMatch[2]).trim();
      logger.debug("Extracted JSON from formatted block");
      return JSON.parse(jsonString);
    }
    
    // If no code block found, try parsing the entire content
    logger.debug("Attempting to parse entire content as JSON");
    return JSON.parse(content.trim());
    
  } catch (error) {
    logger.error(`Failed to parse ${modelName} response as JSON:`, error);
    logger.debug("Raw content:", content);
    throw new Error(`No valid JSON found in ${modelName} response`);
  }
}