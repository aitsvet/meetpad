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

# Configure logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Whisper model
processor = WhisperProcessor.from_pretrained("MikhailMihalis/whisper-large-v3-russian-ties-podlodka-v1.2-ct-int8")
model = WhisperForConditionalGeneration.from_pretrained("MikhailMihalis/whisper-large-v3-russian-ties-podlodka-v1.2-ct-int8")

# Ensure CPU usage
model = model.to('cpu')
model.eval()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message with a button that opens the web app."""
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

async def handle_webapp_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle data from the web app."""
    data = json.loads(update.effective_message.web_app_data.data)
    
    if 'audio_data' in data:
        audio_base64 = data['audio_data']
        await update.message.reply_text("Processing audio...")
        
        try:
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_base64.split(',')[1])
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            
            # Transcribe
            transcription = transcribe_audio(temp_file_path)
            
            # Clean up
            os.unlink(temp_file_path)
            
            await update.message.reply_text(f"Transcription: {transcription}")
            
        except Exception as e:
            logger.error(f"Error processing audio: {e}")
            await update.message.reply_text("Error processing audio. Please try again.")

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using Whisper model."""
    try:
        # Load audio
        audio, sample_rate = sf.read(audio_path)
        
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 16kHz if needed
        if sample_rate != 16000:
            # Simple resampling - just take every nth sample
            ratio = 16000 / sample_rate
            audio = audio[::int(1/ratio)]
        
        # Process with Whisper
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        
        # Generate transcription
        predicted_ids = model.generate(inputs["input_features"])
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription.strip()
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return "Error transcribing audio"

def main() -> None:
    """Start the bot."""
    # Get token from environment variable
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set")
        return
    
    # Create application
    application = Application.builder().token(token).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.StatusUpdate.WEB_APP_DATA, handle_webapp_data))
    
    # Start the bot
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main() 