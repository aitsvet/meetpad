from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        data = request.get_json()
        audio_data = data.get('audio_data')
        
        if not audio_data:
            return jsonify({'error': 'No audio data provided'}), 400
        
        # Send to Telegram bot for processing
        # This will be handled by the bot's webapp_data handler
        return jsonify({'status': 'success', 'message': 'Audio sent to bot for transcription'})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False) 