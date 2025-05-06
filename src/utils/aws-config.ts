import { fromEnv } from "@aws-sdk/credential-providers";

const awsConfig = {
  region: process.env.AWS_REGION || 'us-east-1',
  credentials: fromEnv()
};

export default awsConfig;