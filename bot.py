import os
import logging
import tempfile
import asyncio
import base64
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
import json

# Configure logging to stderr
import sys
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO,
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Initialize Whisper model
logger.info("Loading Whisper processor...")
processor = WhisperProcessor.from_pretrained("MikhailMihalis/whisper-large-v3-russian-ties-podlodka-v1.2-ct-int8")
logger.info("Processor loaded successfully")

logger.info("Loading Whisper model...")
model = WhisperForConditionalGeneration.from_pretrained("MikhailMihalis/whisper-large-v3-russian-ties-podlodka-v1.2-ct-int8")
logger.info("Model loaded successfully")

# Ensure CPU usage
logger.info("Moving model to CPU...")
model = model.to('cpu')
model.eval()
logger.info("Model ready for inference on CPU")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message with a button that opens the web app."""
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    logger.info(f"Start command received from user {user_id} (@{username})")
    
    await update.message.reply_text(
        "Welcome! Use the button below to open the audio transcription app:",
        reply_markup={
            "inline_keyboard": [[
                {
                    "text": "Open Transcription App",
                    "web_app": {"url": "https://aitsvet-meetpad.hf.space"}
                }
            ]]
        }
    )
    logger.info(f"Start message sent to user {user_id}")

async def handle_webapp_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle data from the web app."""
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    logger.info(f"Web app data received from user {user_id} (@{username})")
    
    data = json.loads(update.effective_message.web_app_data.data)
    logger.info(f"Data keys: {list(data.keys())}")
    
    if 'audio_data' in data:
        audio_base64 = data['audio_data']
        audio_size = len(audio_base64)
        logger.info(f"Audio data received, size: {audio_size} characters")
        
        await update.message.reply_text("Processing audio...")
        logger.info(f"Processing started for user {user_id}")
        
        try:
            # Decode base64 audio
            logger.info("Decoding base64 audio...")
            audio_bytes = base64.b64decode(audio_base64.split(',')[1])
            logger.info(f"Audio decoded, bytes: {len(audio_bytes)}")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            logger.info(f"Audio saved to temp file: {temp_file_path}")
            
            # Transcribe
            logger.info("Starting transcription...")
            transcription = transcribe_audio(temp_file_path)
            logger.info(f"Transcription completed: '{transcription[:50]}...'")
            
            # Clean up
            os.unlink(temp_file_path)
            logger.info("Temp file cleaned up")
            
            await update.message.reply_text(f"Transcription: {transcription}")
            logger.info(f"Transcription sent to user {user_id}")
            
        except Exception as e:
            logger.error(f"Error processing audio for user {user_id}: {e}")
            await update.message.reply_text("Error processing audio. Please try again.")
    else:
        logger.warning(f"No audio_data found in web app data from user {user_id}")

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using Whisper model."""
    try:
        logger.info(f"Loading audio from: {audio_path}")
        # Load audio
        audio, sample_rate = sf.read(audio_path)
        logger.info(f"Audio loaded: shape={audio.shape}, sample_rate={sample_rate}")
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            logger.info("Converting stereo to mono")
            audio = audio.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            logger.info(f"Resampling from {sample_rate}Hz to 16000Hz")
            # Simple resampling - just take every nth sample
            ratio = 16000 / sample_rate
            audio = audio[::int(1/ratio)]
            logger.info(f"Resampled audio shape: {audio.shape}")
        
        # Process with Whisper
        logger.info("Processing audio with Whisper processor...")
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        logger.info("Audio processed, generating transcription...")
        
        # Generate transcription
        predicted_ids = model.generate(inputs["input_features"])
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        logger.info(f"Raw transcription: '{transcription}'")
        
        return transcription.strip()
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return "Error transcribing audio"

def main() -> None:
    """Start the bot."""
    logger.info("Starting Telegram bot...")
    
    # Get token from environment variable
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set")
        return
    
    logger.info("Bot token found, creating application...")
    
    # Create application
    application = Application.builder().token(token).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.StatusUpdate.WEB_APP_DATA, handle_webapp_data))
    logger.info("Handlers registered")
    
    # Start the bot
    logger.info("Starting bot polling...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 