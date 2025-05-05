import "dotenv/config";
import * as path from "path";
import * as fs from "fs";
import DeepRacerAgent, { LogLevel } from "./deepracer-agent";
import { Logger, getLogger } from "./utils/logger";
import { parseCommandLineArgs } from "./utils/command-line-parser";

/**
 * Main entry point for the DeepRacer LLM Agent application
 */
async function main() {
  const mainLogger = getLogger("Main");

  // Parse command line arguments
  const args = process.argv.slice(2);
  const options = parseCommandLineArgs(args, mainLogger);
  
  // If options is null, either help was requested or there was an error
  if (options === null) {
    return;
  }

  // Default skipFactor to 2 if not specified
  const skipFactor = options.skipFactor || 2;
  mainLogger.info(`Using frame skip factor: ${skipFactor}`);

  // Create the DeepRacer agent with a specified log level
  const logLevel = process.env.LOG_LEVEL
    ? LogLevel[process.env.LOG_LEVEL.toUpperCase() as keyof typeof LogLevel] ||
      LogLevel.INFO
    : LogLevel.INFO;
  const agent = new DeepRacerAgent({
    logLevel,
    maxContextMessages: options.maxContextMessages,
  });

  try {
    const testImagesDir = path.join(__dirname, "..", "test-images");

    if (!fs.existsSync(testImagesDir)) {
      mainLogger.error(`Test images directory not found: ${testImagesDir}`);
      return;
    }

    // Get all image files (jpg, jpeg, png) and sort them numerically
    const imageFiles = fs
      .readdirSync(testImagesDir)
      .filter((file) => /\.(jpg|jpeg|png)$/i.test(file))
      .sort((a, b) => {
        // Extract numbers from filenames for proper numeric sorting
        const numA = parseInt((a.match(/\d+/) || ["0"])[0]);
        const numB = parseInt((b.match(/\d+/) || ["0"])[0]);
        return numA - numB;
      });

    if (imageFiles.length === 0) {
      mainLogger.error("No image files found in test-images directory");
      return;
    }

    mainLogger.info(`Found ${imageFiles.length} images to process`);

    // Apply start offset if specified
    const startOffset = options.startOffset || 0;
    if (startOffset > 0) {
      mainLogger.info(
        `Starting from image ${startOffset} (skipping ${startOffset} images)`
      );
    }

    // Determine how many frames to process
    const maxFrames =
      options.frames ||
      Math.floor((imageFiles.length - startOffset) / skipFactor);
    const framesToProcess = Math.min(
      maxFrames,
      Math.floor((imageFiles.length - startOffset) / skipFactor)
    );
    mainLogger.info(
      `Will process ${framesToProcess} frames (every ${skipFactor}th frame)`
    );

    // Process each image in sequence with the specified skip factor
    for (let i = 0; i < framesToProcess; i++) {
      const frameIndex = startOffset + i * skipFactor;
      const imagePath = path.join(testImagesDir, imageFiles[frameIndex]);
      mainLogger.info(
        `\n[${i + 1}/${framesToProcess}] 🏎️ Processing image: ${
          imageFiles[frameIndex]
        }`
      );

      const action = await agent.processImageFile(imagePath);
      mainLogger.info("Recommended action:", JSON.stringify(action, null, 2));

      // Optional: Add a small delay between processing to avoid rate limits
      if (i < framesToProcess - 1) {
        mainLogger.debug("Waiting before processing next image...");
        await new Promise((resolve) => setTimeout(resolve, 50));
      }
    }

    // Log the total token usage with accurate pricing
    const tokenUsage = agent.getTokenUsage();
    mainLogger.info("\n📈 Token Usage Summary:");
    mainLogger.info(
      `   Prompt tokens:     ${tokenUsage.promptTokens.toLocaleString()}`
    );
    mainLogger.info(
      `   Completion tokens: ${tokenUsage.completionTokens.toLocaleString()}`
    );
    mainLogger.info(
      `   Total tokens:      ${tokenUsage.totalTokens.toLocaleString()}`
    );

    // Display pricing rates used for calculation
    mainLogger.info(
      `   Prompt rate:       $${tokenUsage.pricing.promptRate.toFixed(
        4
      )}/1K tokens`
    );
    mainLogger.info(
      `   Completion rate:   $${tokenUsage.pricing.completionRate.toFixed(
        4
      )}/1K tokens`
    );
    mainLogger.info(
      `   Estimated cost:    $${tokenUsage.estimatedCost.toFixed(4)}`
    );

    mainLogger.info("\n✅ All images processed successfully");
  } catch (error) {
    mainLogger.error("❌ Error processing images:", error);
  }
}

// Run the main function if this script is executed directly
if (require.main === module) {
  main().catch(console.error);
}

// Export the main function for potential programmatic use
export default main;