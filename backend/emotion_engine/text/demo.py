import os
import sys
import time
import logging
from pathlib import Path

# Setup path manipulation so the script can be run directly from any directory
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.emotion_engine.text.inference import TextEmotionEngine
from backend.emotion_engine.text.schemas import TextEmotionResponse

# Setup logger for startup and system information
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_engine() -> TextEmotionEngine:
    """
    Initializes and loads the text emotion engine, measuring loading time.
    """
    logger.info("Initializing Text Emotion Engine...")
    start_time = time.perf_counter()
    
    engine = TextEmotionEngine()
    engine.load_model()
    
    elapsed_time = time.perf_counter() - start_time
    logger.info(f"Engine loaded successfully in {elapsed_time:.2f} seconds.")
    return engine


def print_results(input_text: str, response: TextEmotionResponse, inference_time_ms: float) -> None:
    """
    Prints the emotion prediction results in a clean, standardized format.
    """
    print("==================================================")
    print("Input:")
    print(input_text)
    print()
    print(f"Primary Emotion : {response.primary_emotion}")
    print(f"Confidence      : {response.confidence:.4f}")
    print()
    print("Emotion Probabilities")
    print("----------------------------------")
    for emotion, prob in response.probabilities.items():
        print(f"{emotion.capitalize():<11}: {prob:.4f}")
    print(f"Inference Time  : {inference_time_ms:.2f} ms")
    print("==================================================")


def run_interactive_loop(engine: TextEmotionEngine) -> None:
    """
    Runs the interactive CLI loop for manual text emotion testing.
    """
    print("\nInteractive Text Emotion Engine Demo")
    print("Type 'exit' or 'quit' to exit.\n")
    
    while True:
        try:
            user_input = input("Enter text to analyze: ")
            stripped = user_input.strip()
            
            if stripped.lower() in ("exit", "quit"):
                print("Exiting demo. Goodbye!")
                break
                
            if not stripped:
                print("Input is empty. Please enter some text.\n")
                continue
                
            # Measure inference time in milliseconds
            start_time = time.perf_counter()
            response = engine.predict(stripped)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            
            print_results(stripped, response, elapsed_ms)
            print()
            
        except KeyboardInterrupt:
            print("\nExiting demo. Goodbye!")
            break
        except Exception as e:
            logger.error(f"An error occurred during prediction: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    try:
        engine = load_engine()
        run_interactive_loop(engine)
    except Exception as e:
        logger.critical(f"Failed to start the demo: {e}")
        sys.exit(1)
