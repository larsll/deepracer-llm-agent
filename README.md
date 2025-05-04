# DeepRacer LLM Agent

This project implements a demonstrator agent based on Large Language Models (LLMs) for AWS DeepRacer. It enables intelligent decision-making for autonomous racing by leveraging language models to process visual inputs and generate optimal racing strategies.

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
npm agent
```

The LLM agent will process track images from DeepRacer and make racing decisions. You can configure the agent parameters in the configuration files.

## Features

- Natural language processing for DeepRacer control using LLMs
- Integration with AWS Bedrock for efficient model inferencing
- Computer vision capabilities for track and obstacle detection
- Adaptive decision-making based on race conditions
- Reinforcement learning components to improve over time
- Performance analytics and visualization tools
- Customizable prompting strategies for different racing scenarios

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.