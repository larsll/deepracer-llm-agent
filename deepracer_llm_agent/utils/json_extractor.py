import json
import logging
import re
from typing import Any, TypeVar, Optional

T = TypeVar('T')


def extract_json_from_llm_response(
    content: str,
    logger: Optional[logging.Logger] = None,
    model_name: str = "LLM"
) -> Any:
    """
    Extracts and parses JSON from LLM text responses.
    Handles various formats including code blocks and direct JSON.

    Args:
        content: The text content from an LLM response
        logger: Logger instance for debugging (optional)
        model_name: Optional model name for better error messages

    Returns:
        Parsed JSON object

    Raises:
        ValueError: If no valid JSON could be found in the response
    """
    # Use a default logger if none provided
    if logger is None:
        logger = logging.getLogger("JsonExtractor")

    logger.debug(
        f"Raw content from {model_name} model: {content[:200]}"
    )

    # Try to extract JSON from content - either from code blocks or directly
    try:
        # First try to find JSON in code blocks
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```|(\{[\s\S]*?\})"
        json_match = re.search(json_pattern, content)

        if json_match:
            # Parse JSON from code block
            json_string = (json_match.group(1) or json_match.group(2)).strip()
            logger.debug("Extracted JSON from formatted block")
            return json.loads(json_string)

        # If no code block found, try parsing the entire content
        logger.debug("Attempting to parse entire content as JSON")
        return json.loads(content.strip())

    except (json.JSONDecodeError, AttributeError) as error:
        logger.error(f"Failed to parse {model_name} response as JSON: {error}")
        logger.debug(f"Raw content: {content}")
        raise ValueError(f"No valid JSON found in {model_name} response")
