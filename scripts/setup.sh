#!/bin/bash

# This script automates the setup process for the bedrock-image-processing project.

# Install dependencies
echo "Installing dependencies..."
npm install

# Copy the example environment file to .env
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
else
    echo ".env file already exists. Skipping copy."
fi

# Set up environment variables
echo "Setting up environment variables..."
export $(cat .env | xargs)

echo "Setup complete!"