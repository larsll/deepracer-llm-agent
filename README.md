# Bedrock Image Processing

This project provides a set of tools for processing images and interacting with Amazon Bedrock. It includes functionality for uploading images, processing them, and handling responses from the Bedrock service.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/bedrock-image-processing.git
   ```
2. Navigate to the project directory:
   ```
   cd bedrock-image-processing
   ```
3. Install the dependencies:
   ```
   npm install
   ```
4. Set up your environment variables by copying the `.env.example` to `.env` and filling in your AWS credentials:
   ```
   cp .env.example .env
   ```

## Usage

To start the application, run:
```
npm start
```

You can use the provided services to upload and process images. Refer to the documentation in the `src/services/bedrock.ts` file for more details on available methods.

## Features

- Upload images to Amazon Bedrock
- Process images with various utility functions
- Resize and convert image formats before uploading

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.