from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import base64
import tempfile
import logging
from faster_whisper import WhisperModel, BatchedInferencePipeline
import requests
import json

# Configure logging to stderr
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize faster-whisper model
logger.info("Loading faster-whisper model...")
model = WhisperModel(
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    device="cpu",
    compute_type="int8"
)
batched_model = BatchedInferencePipeline(model=model)
logger.info("Model loaded successfully and ready for inference on CPU")

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using faster-whisper model."""
    try:
        logger.info(f"Transcribing audio from: {audio_path}")
        
        # Transcribe with faster-whisper
        segments, info = batched_model.transcribe(audio_path, multilingual=True, language="ru", batch_size=16,
                                                  vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
        logger.info(f"Transcription info: language={info.language}, language_probability={info.language_probability:.2f}")
        
        # Collect all segments
        transcription_parts = []
        for segment in segments:
            transcription_parts.append(segment.text)
            logger.info(f"Segment: '{segment.text}'")
        
        transcription = " ".join(transcription_parts).strip()        
        return transcription
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return ""

def send_to_telegram(chat_id: str, text: str, bot_token: str) -> bool:
    """Send transcribed text to Telegram chat."""
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            logger.info(f"Successfully sent transcription to chat {chat_id}")
            return True
        else:
            logger.error(f"Failed to send message to Telegram: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error sending to Telegram: {e}")
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        data = request.get_json()
        audio_data = data.get('audio_data')
        user_id = data.get('user_id', 'unknown')
        chat_id = data.get('chat_id', None)
        
        logger.info(f"Transcription request received from user {user_id}")
        
        if not audio_data:
            logger.error("No audio data provided")
            return jsonify({'error': 'No audio data provided'}), 400
        
        logger.info("Decoding base64 audio...")
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
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
        
        # Send to Telegram if chat_id is provided
        sent_to_telegram = False
        if chat_id:
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            if bot_token:
                sent_to_telegram = send_to_telegram(chat_id, f"{transcription}", bot_token)
            else:
                logger.warning("TELEGRAM_BOT_TOKEN not found in environment variables")
        
        return jsonify({
            'status': 'success', 
            'transcription': transcription,
            'sent_to_telegram': sent_to_telegram,
            'message': 'Audio transcribed successfully' + (' and sent to Telegram' if sent_to_telegram else '')
        })
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/webhook', methods=['POST'])
def telegram_webhook():
    """Handle Telegram webhook for sending messages."""
    try:
        data = request.get_json()
        logger.info(f"Received webhook data: {json.dumps(data, indent=2)}")
        
        # Extract message data
        if 'message' in data:
            chat_id = data['message']['chat']['id']
            text = data['message'].get('text', '')
            
            if text.startswith('/start'):
                # Send welcome message with web app button
                welcome_text = "🎤 <b>Welcome to Audio Transcription Bot!</b>\n\n"
                welcome_text += "Click the button below to open the transcription app:\n\n"
                welcome_text += "📱 <b>Transcription App</b>"
                
                bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
                if bot_token:
                    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                    payload = {
                        "chat_id": chat_id,
                        "text": welcome_text,
                        "parse_mode": "HTML",
                        "reply_markup": {
                            "inline_keyboard": [[
                                {
                                    "text": "🎤 Open Transcription App",
                                    "web_app": {"url": os.getenv('WEBAPP_URL', 'https://aitsvet-meetpad.hf.space')}
                                }
                            ]]
                        }
                    }
                    response = requests.post(url, json=payload, timeout=10)
                    logger.info(f"Webhook response: {response.status_code}")
        
        return jsonify({'status': 'ok'})
        
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=7860, debug=False) 