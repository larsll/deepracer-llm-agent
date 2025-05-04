import 'dotenv/config';
import { BedrockService } from './services/bedrock';

const init = async () => {
    // Load environment variables
    const awsAccessKeyId = process.env.AWS_ACCESS_KEY_ID;
    const awsSecretAccessKey = process.env.AWS_SECRET_ACCESS_KEY;
    const awsRegion = process.env.AWS_REGION;

    if (!awsAccessKeyId || !awsSecretAccessKey || !awsRegion) {
        throw new Error('Missing AWS configuration in environment variables');
    }

    // Initialize Bedrock service
    const bedrockService = new BedrockService(awsAccessKeyId, awsSecretAccessKey, awsRegion);

    // Additional setup can be done here
};

init().catch(error => {
    console.error('Error initializing application:', error);
});