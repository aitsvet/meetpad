from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import base64
import tempfile
import logging
from faster_whisper import WhisperModel, BatchedInferencePipeline

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
        segments, info = batched_model.transcribe(audio_path, batch_size=16, without_timestamps=True, multilingual=True, language="ru")
        
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
        return "Error transcribing audio"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        data = request.get_json()
        audio_data = data.get('audio_data')
        user_id = data.get('user_id', 'unknown')
        
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
        
        return jsonify({
            'status': 'success', 
            'transcription': transcription,
            'message': 'Audio transcribed successfully'
        })
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=7860, debug=False) 