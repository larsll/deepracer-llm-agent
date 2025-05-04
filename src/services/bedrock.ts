import { BedrockRuntimeClient, InvokeModelCommand } from "@aws-sdk/client-bedrock-runtime";
import * as fs from 'fs';

class BedrockService {
    private bedrockClient: BedrockRuntimeClient;
    
    constructor() {
        // Use credentials from .env file (loaded by dotenv)
        this.bedrockClient = new BedrockRuntimeClient({
            region: process.env.AWS_REGION,
            credentials: {
                accessKeyId: process.env.AWS_ACCESS_KEY_ID as string,
                secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY as string
            }
        });
        
        console.log(`🔑 Initialized Bedrock client for region: ${process.env.AWS_REGION}`);
    }

    /**
     * Process an image directly with Bedrock using base64 encoding
     * @param imageBuffer The image as a buffer
     * @param modelId The Bedrock model ID to use
     * @param prompt Instructions for the model
     */
    async processImageSync(
        imageBuffer: Buffer, 
        modelId: string = process.env.DEFAULT_MODEL_ID || "anthropic.claude-3-sonnet-20240229-v1:0", 
        prompt: string = "Analyze this image"
    ): Promise<any> {
        // Convert image to base64
        const base64Image = imageBuffer.toString('base64');
        
        // Create request payload based on the model
        const payload = this.createPayloadForModel(modelId, base64Image, prompt);
        
        // Invoke Bedrock model with timeout
        const command = new InvokeModelCommand({
            modelId,
            contentType: "application/json",
            accept: "application/json",
            body: JSON.stringify(payload)
        });

        try {
            console.log(`🔄 Processing image with model: ${modelId}`);
            const timeout = parseInt(process.env.TIMEOUT_MS || '30000');
            
            // Create a promise with timeout
            const timeoutPromise = new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Request timed out')), timeout)
            );
            
            // Race between the actual request and the timeout
            const response = await Promise.race([
                this.bedrockClient.send(command),
                timeoutPromise
            ]) as any;
            
            // Parse the response body
            const responseBody = JSON.parse(new TextDecoder().decode(response.body));
            return responseBody;
        } catch (error) {
            console.error("❌ Error processing image with Bedrock:", error);
            throw error;
        }
    }

    /**
     * Process an image from a file path
     */
    async processImageFromFile(
        filePath: string,
        modelId: string = "amazon.titan-image-generator-v1",
        prompt: string = "Analyze this image"
    ): Promise<any> {
        const imageBuffer = fs.readFileSync(filePath);
        return this.processImageSync(imageBuffer, modelId, prompt);
    }

    /**
     * Create the appropriate payload structure based on the model
     */
    private createPayloadForModel(modelId: string, base64Image: string, prompt: string): any {
        // Different models may require different payload structures
        if (modelId.includes('claude')) {
            // Claude models payload
            return {
                anthropic_version: "bedrock-2023-05-31",
                messages: [
                    {
                        role: "user",
                        content: [
                            { type: "text", text: prompt },
                            { type: "image", source: { type: "base64", media_type: "image/jpeg", data: base64Image } }
                        ]
                    }
                ],
                max_tokens: 1000
            };
        } else if (modelId.includes('mistral.pixtral')) {
            // Mistral Pixtral models payload - using the exact format they require
            return {
                messages: [
                    {
                        role: "user",
                        content: [
                            { type: "text", text: prompt },
                            { type: "image_url", image_url: { url: `data:image/jpeg;base64,${base64Image}` } }
                        ]
                    }
                ]
            };
        } else if (modelId.includes('titan-image')) {
            // Amazon Titan Image Generator payload
            return {
                taskType: "IMAGE_ANALYSIS",
                imageData: base64Image,
                prompt
            };
        } else {
            // Default payload structure
            return {
                prompt,
                image_data: base64Image
            };
        }
    }
}

export default BedrockService;