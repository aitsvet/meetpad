from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import base64
import tempfile
import logging
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# Configure logging to stderr
import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

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

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using Whisper model."""
    try:
        logger.info(f"Loading audio from: {audio_path}")
        import soundfile as sf
        
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