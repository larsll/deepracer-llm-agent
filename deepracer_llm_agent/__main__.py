import argparse
import os
import sys
import glob
import time
import json
import logging
from pathlib import Path
from .agent import LLMAgent
from typing import Dict, Any, Optional


def setup_logger(log_level_name: str = "INFO") -> logging.Logger:
    """Set up and return a logger for the main module"""
    log_level = getattr(logging, log_level_name.upper(), logging.INFO)
    logger = logging.getLogger("Main")
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def parse_arguments() -> Optional[Dict[str, Any]]:
    """Parse command line arguments and return options dictionary"""
    parser = argparse.ArgumentParser(description='DeepRacer LLM Agent')
    parser.add_argument('--frames', '-f', type=int,
                        help='Number of frames to process')
    parser.add_argument('--skip', '-x', type=int, default=2,
                        help='Process every Nth frame')
    parser.add_argument('--start', '-s', type=int,
                        default=0, help='Start from Nth image')
    parser.add_argument('--config', '-c', type=str,
                        default='model_metadata.json', help='Path to metadata file')
    parser.add_argument('--images', '-i', type=str,
                        default='./test-images', help='Path to folder with images')
    
    if len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help']:
        parser.print_help()
        return None

    return vars(parser.parse_args())


def main():
    """Main entry point for the DeepRacer LLM Agent application"""
    # Set up logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger = setup_logger(log_level)

    # Parse command line arguments
    options = parse_arguments()
    if options is None:
        return

    # Extract options with defaults
    skip_factor = options.get('skip', 2)
    logger.debug(f"Using frame skip factor: {skip_factor}")

    # Create the DeepRacer agent
    try:
        agent = LLMAgent(options.get('config', 'model_metadata.json'))

        # Get the directory with test images
        if os.path.isabs(options['images']):
            test_images_dir = Path(options['images'])
        else:
            test_images_dir = Path(Path.cwd(), options['images'])

        if not os.path.exists(test_images_dir):
            logger.error(f"Test images directory not found: {test_images_dir}")
            return

        # Get all image files and sort them numerically
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_files.extend(glob.glob(str(test_images_dir / ext)))

        # Sort files numerically based on numbers in filenames
        image_files.sort(key=lambda f: int(
            ''.join(filter(str.isdigit, os.path.basename(f))) or 0))

        if not image_files:
            logger.error("No image files found in test-images directory")
            return

        start_offset = options.get('start', 0)
        max_frames = options.get('frames') or (
            (len(image_files) - start_offset) // skip_factor)
        frames_to_process = min(
            max_frames, (len(image_files) - start_offset) // skip_factor)

        logger.info(
            f"🖼️ Found {len(image_files)} images. Starting from image {start_offset}. "
            f"Will process {frames_to_process} frames (every {skip_factor}. frame)."
        )

        # Process each image in sequence with the specified skip factor
        for i in range(frames_to_process):
            frame_index = start_offset + i * skip_factor
            image_path = image_files[frame_index]
            image_name = os.path.basename(image_path)
            logger.info(
                f"[{i + 1}/{frames_to_process}] 🏎️ Processing image: {image_name}")

            action = agent.process_image(image_path)
            logger.info(f"🛞 Action to take: {json.dumps(action, indent=2)}")

            # Optional delay between processing to avoid rate limits
            if i < frames_to_process - 1:
                logger.debug("Waiting before processing next image...")
                time.sleep(0.05)

        # Log the total token usage with pricing
        token_usage = agent.get_token_usage()
        if token_usage:
            logger.info("\n📈 Token Usage Summary:")
            logger.info(
                f"   Prompt tokens:     {token_usage.get('prompt_tokens', 0):,}")
            logger.info(
                f"   Completion tokens: {token_usage.get('completion_tokens', 0):,}")
            logger.info(
                f"   Total tokens:      {token_usage.get('total_tokens', 0):,}")

            # Display pricing rates if available
            pricing = token_usage.get('pricing', {})
            if pricing:
                logger.info(
                    f"   Prompt rate:       ${pricing.get('prompt_rate', 0):.4f}/1K tokens")
                logger.info(
                    f"   Completion rate:   ${pricing.get('completion_rate', 0):.4f}/1K tokens")
                logger.info(
                    f"   Estimated cost:    ${token_usage.get('estimated_cost', 0):.4f}")

        logger.info("\n✅ All images processed successfully")

    except Exception as e:
        logger.error(f"❌ Error processing images: {e}", exc_info=True)


if __name__ == "__main__":
    main()
