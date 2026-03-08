#!/usr/bin/env python3
"""
Coqui TTS Voice Generation Script
Generates voice-over audio from text
"""

import argparse
from pathlib import Path
import logging
import torch
from TTS.api import TTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_voiceover(text, output_path, speaker="default"):
    """Generate voice-over using Coqui TTS"""
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use GPU if available
    gpu = torch.cuda.is_available()
    logger.info(f"Using GPU: {gpu}")
    
    try:
        # Initialize TTS
        logger.info("Initializing Coqui TTS...")
        tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=gpu)
        
        logger.info(f"Generating speech for: {text}")
        
        # Generate speech
        tts.tts_to_file(
            text=text,
            file_path=str(output_path),
            speaker=speaker,
            verbose=True,
            emotion="Happy"
        )
        
        logger.info(f"Voice-over saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error generating voice-over: {str(e)}")
        # Try with GPU disabled if it fails
        if gpu:
            logger.info("Retrying without GPU...")
            tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", gpu=False)
            tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                verbose=True,
                emotion="Happy"
            )
            logger.info(f"Voice-over saved to: {output_path}")
        else:
            raise

def main():
    parser = argparse.ArgumentParser(description="Generate voice-over with Coqui TTS")
    parser.add_argument("--text", required=True, help="Text to convert to speech")
    parser.add_argument("--output", required=True, help="Output audio file path")
    parser.add_argument("--speaker", default="default", help="Speaker name/ID")
    
    args = parser.parse_args()
    
    try:
        generate_voiceover(args.text, args.output, args.speaker)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise

if __name__ == "__main__":
    main()