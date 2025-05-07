# DeepRacer LLM Agent

This code implements a DeepRacer agent that processes track images using Large Language Models (LLMs) through AWS Bedrock. The agent takes camera images from a DeepRacer car, converts them to base64, and sends them to an LLM (Claude, Mistral, or Nova) along with a structured prompt and action space information. For each image, the LLM receives: 
1) a system prompt describing its role as a driving assistant, 
2) the action space configuration (continuous or discrete steering/speed ranges), 
3) the camera image itself, and 
4) a request to analyze the image and make driving decisions. 

The LLM responds with JSON containing recommended steering angle and speed values, along with reasoning for its decision. The code handles all AWS service integration, maintains conversation context between frames, validates responses, and tracks token usage and costs. It's essentially creating a vision-language model based autonomous driving agent that can explain its decisions.

## Installation

### Local Setup

1. Clone the repository:
   ```
   git clone https://github.com/larsll/deepracer-llm-agent.git
   ```
2. Navigate to the project directory:
   ```
   cd deepracer-llm-agent
   ```
3. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
5. Set up your environment variables by copying the `.env.example` to `.env` and filling in your AWS credentials:
   ```
   cp .env.example .env
   ```
   ```

### AWS Configuration

The setup requires two things:
 1. AWS credentials that allows to execute Bedrock API calls.
 2. Enabled models (currently Claude, Mistral and Nova are supported) in the right region.

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