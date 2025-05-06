# DeepRacer LLM Agent

This project implements a demonstrator agent based on Large Language Models (LLMs) for AWS DeepRacer. It enables intelligent decision-making for autonomous racing by leveraging language models to process visual inputs and generate optimal racing strategies.

## Features

- Natural language processing for DeepRacer control using LLMs
- Integration with AWS Bedrock for efficient model inferencing
- Computer vision capabilities for track and obstacle detection
- Adaptive decision-making based on race conditions
- Configurable context window to provide the LLM with prior knowledge

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/larsll/deepracer-llm-agent.git
   ```
2. Navigate to the project directory:
   ```
   cd deepracer-llm-agent
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

To start the agent, run:
```
npm agent -- <options>
```

### Command Line Options

The agent can be configured using the following command-line options:

- `--frames`, `-f <number>`: Specify the number of frames to process (default: process all frames).
- `--speed`, `-x <number>`: Process every Nth frame (default: 2).
- `--start`, `-s <number>`: Start processing from the Nth image (default: 0).
- `--config`, `-c <file>`: Provide the path to the metadata file.
- `--help`, `-h`: Display this help message.

The LLM agent will process track images from DeepRacer (the examples are in `test-images/`) and make racing decisions. You can configure the agent parameters in 
an adjusted `model_metadata.json` - see examples in `examples/`.


## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.