import 'dotenv/config';
import * as path from 'path';
import * as fs from 'fs';
import BedrockService from './services/bedrock';

const init = async () => {
    // Validate essential environment variables from .env
    const requiredEnvVars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION'];
    const missingVars = requiredEnvVars.filter(varName => !process.env[varName]);
    
    if (missingVars.length > 0) {
        console.error(`❌ Missing required environment variables: ${missingVars.join(', ')}`);
        console.error('Please update your .env file with the required values');
        process.exit(1);
    }
    
    // Initialize the Bedrock service
    const bedrockService = new BedrockService();
    console.log('🚀 Bedrock Image Processing Service initialized successfully');
    
    // Example test
    try {
        const testImagePath = path.join(__dirname, '..', 'test-images', '001832_resized.jpg');
        if (fs.existsSync(testImagePath)) {
            console.log('🧪 Running test with sample image...');
            
            // Use inference profile if available, otherwise fallback to default model
            const modelIdOrArn = process.env.INFERENCE_PROFILE_ARN || 
                                 process.env.DEFAULT_MODEL_ID || 
                                'anthropic.claude-3-sonnet-20240229-v1:0';
            
            console.log(`Using model/inference profile: ${modelIdOrArn}`);
            
            const result = await bedrockService.processImageFromFile(
                testImagePath,
                modelIdOrArn,
                'Describe this image in detail'
            );
            console.log('✅ Test successful!');
            console.log('📝 Result:', JSON.stringify(result, null, 2));
        } else {
            console.error(`❌ Test image not found at: ${testImagePath}`);
        }
    } catch (error) {
        console.error('❌ Test failed:', error);
        if (error instanceof Error) {
            console.error(error.message);
            console.error(error.stack);
        }
    }
};

init().catch(error => {
    console.error('❌ Error initializing application:', error);
    process.exit(1);
});