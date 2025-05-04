#!/bin/bash
set -e

echo "🔧 Setting up Bedrock Image Processing project..."

# Check if .env file exists, if not create from example
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
    echo "⚠️ Please edit .env file with your AWS credentials"
fi

# Create directories for testing
mkdir -p test-images
mkdir -p outputs

# Install dependencies
echo "📦 Installing dependencies..."
npm install

echo "✅ Setup complete!"
echo "🚀 To run the project, use: npm start"
echo "⚠️ Don't forget to update your AWS credentials in the .env file"