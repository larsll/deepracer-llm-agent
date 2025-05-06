import { Logger, getLogger } from "./logger";

/**
 * Options for the DeepRacer LLM Agent
 */
export interface CommandLineOptions {
  frames?: number;
  skipFactor?: number;
  startOffset?: number;
  metadataFilePath?: string;
}

/**
 * Parse command line arguments for the DeepRacer LLM Agent
 * @param args Command line arguments array (typically process.argv.slice(2))
 * @param logger Optional logger for error messages
 * @returns Parsed options object or null if invalid arguments or help requested
 */
export function parseCommandLineArgs(
  args: string[],
  logger?: Logger
): CommandLineOptions | null {
  const options: CommandLineOptions = {};
  const log = logger || getLogger("CommandLineParser");

  // Process command line arguments
  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--frames" || args[i] === "-f") {
      options.frames = parseInt(args[++i], 10);
      if (isNaN(options.frames) || options.frames <= 0) {
        log.error("Invalid value for --frames. Must be a positive integer.");
        return null;
      }
    } else if (args[i] === "--speed" || args[i] === "-x") {
      options.skipFactor = parseInt(args[++i], 10);
      if (isNaN(options.skipFactor) || options.skipFactor <= 0) {
        log.error("Invalid value for --speed. Must be a positive integer.");
        return null;
      }
    } else if (args[i] === "--start" || args[i] === "-s") {
      options.startOffset = parseInt(args[++i], 10);
      if (isNaN(options.startOffset) || options.startOffset < 0) {
        log.error("Invalid value for --start. Must be a non-negative integer.");
        return null;
      }
    } else if (args[i] === "--config" || args[i] === "-c") {
      options.metadataFilePath = args[++i].trim();
      if (
        typeof options.metadataFilePath !== "string" ||
        options.metadataFilePath.length === 0
      ) {
        log.error("Invalid value for --config. Must be a non-empty string.");
        return null;
      }
    } else if (args[i] === "--help" || args[i] === "-h") {
      printHelp();
      return null;
    }
  }

  return options;
}

/**
 * Print help information for the command line interface
 */
export function printHelp(): void {
  console.log(`
DeepRacer LLM Agent Image Processing

Options:
  --frames, -f <number>   Number of frames to process (default: process all frames)
  --speed, -x <number>    Process every Nth frame (default: 2)
  --start, -s <number>    Start processing from Nth image (default: 0)
  --config, -c <file>     The path to the metadata file
  --help, -h              Show this help message
  `);
}
