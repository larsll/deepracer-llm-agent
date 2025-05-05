/**
 * Interface for token usage data
 */
export interface TokenUsageData {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

/**
 * Interface for model-specific handlers
 */
export interface IModelHandler {
  /**
   * Get the type of model this handler is for
   */
  getModelType(): string;
  
  /**
   * Set system prompt for the model
   */
  setSystemPrompt(prompt: string): void;
  
  /**
   * Set max context messages to retain
   */
  setMaxContextMessages(max: number): void;
  
  /**
   * Clear conversation history
   */
  clearConversation(): void;
  
  /**
   * Create the API payload specific to this model type
   */
  createPayload(base64Image: string, prompt: string): any;
  
  /**
   * Process and parse the response from the API
   */
  processResponse(response: any, prompt: string, base64Image: string): any;
  
  /**
   * Extract driving action from response
   */
  extractDrivingAction(response: any): any;
  
  /**
   * Extract token usage from response in model-specific format
   */
  extractTokenUsage(response: any): TokenUsageData;
}

export interface Message {
  role: string;
  content: any;
}

export interface DrivingAction {
  speed: number;
  steering_angle: number;
  reasoning?: string;
  knowledge?: string;
  fallback?: boolean;
  error?: string;
}