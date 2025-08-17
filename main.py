from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import base64
import logging
from faster_whisper import WhisperModel, BatchedInferencePipeline
import requests
import json
import torch
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import sys
import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder=os.path.dirname(os.path.abspath(__file__)))
CORS(app)

# Load prompt templates
with open('agenda.txt', 'r', encoding='utf-8') as file:
    AGENDA_SYSTEM_PROMPT = file.read()

with open('chunk.txt', 'r', encoding='utf-8') as file:
    CHUNK_PROMPT_TEMPLATE = file.read()

# In-memory storage for user context
user_context = {}

# Initialize faster-whisper model
logger.info("Loading faster-whisper model...")
model = WhisperModel(
    "deepdml/faster-whisper-large-v3-turbo-ct2",
    device="cuda" if torch.cuda.is_available() else "cpu",
    compute_type="int8_float16" if torch.cuda.is_available() else "int8"
)
batched_model = BatchedInferencePipeline(model=model)
logger.info("Model loaded successfully")

# Telegram bot setup
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
WEBAPP_URL = os.getenv('WEBAPP_URL', 'https://meetpad.aitsvet.ru')
OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://openrouter.ai/api/v1')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'openai/gpt-oss-20b:free')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if TELEGRAM_BOT_TOKEN:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
else:
    application = None
    logger.warning("TELEGRAM_BOT_TOKEN not set. Bot will not run.")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = "🎤 <b>Welcome to Audio Transcription Bot!</b>\n"
    welcome_text += "Click below to open the app:\n"
    await update.message.reply_text(
        welcome_text,
        parse_mode='HTML',
        reply_markup={
            "inline_keyboard": [[
                {
                    "text": "🎤 Open Transcription App",
                    "web_app": {"url": WEBAPP_URL}
                }
            ]]
        }
    )

async def handle_message_or_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.message.chat_id)
    raw_text = None

    if update.message.text:
        raw_text = update.message.text
    elif update.message.document:
        doc = update.message.document
        file_name = doc.file_name.lower()
        if not any(file_name.endswith(ext) for ext in ['.txt', '.md', '.csv']):
            await update.message.reply_text("❌ I can only process .txt, .md, and .csv files.")
            return

        await update.message.reply_text("📄 Receiving file...")
        try:
            file = await context.bot.get_file(doc.file_id)
            response = requests.get(file.file_path)
            response.raise_for_status()

            if file_name.endswith('.csv'):
                import io
                import csv
                reader = csv.reader(io.StringIO(response.text))
                raw_text = " ".join([" ".join(row) for row in reader])
            else:
                raw_text = response.text
        except Exception as e:
            logger.error(f"Error processing document: {e}", exc_info=True)
            await update.message.reply_text("❌ Sorry, there was an error processing your file.")
            return

    if not raw_text or not raw_text.strip():
        await update.message.reply_text("Please provide valid text or a supported file.")
        return

    await update.message.reply_text("🔄 Processing your input...")

    try:
        # First: extract agenda info only once per chat
        if chat_id not in user_context:
            agenda_result = process_with_gpt(
                system_prompt=AGENDA_SYSTEM_PROMPT,
                user_text=raw_text,
                use_json=True
            )
            parsed_agenda_result = json.loads(agenda_result)
            logger.info(f"Parsed agenda result: {parsed_agenda_result}")
            questions_str = ""
            for (i, q) in enumerate(parsed_agenda_result["questions"]):
                questions_str += f"{i+1}. {q}\n"
            user_context[chat_id] = {
                "duration_minutes": parsed_agenda_result["duration_minutes"],
                "start_time": datetime.datetime.now().timestamp(), 
                "chunk_system_prompt": CHUNK_PROMPT_TEMPLATE.format(
                    questions=questions_str,
                    duration_minutes=parsed_agenda_result["duration_minutes"]
                ),
            }

        await update.message.reply_text(
            f"```\n{json.dumps(user_context[chat_id], indent=2, ensure_ascii=False)}\n```",
            parse_mode='Markdown'
        )
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        await update.message.reply_text("❌ Sorry, an error occurred during processing.")

json_block = lambda s: s[s.find('{'):s.rfind('}')+1] if s.find('{') != -1 and s.rfind('}') != -1 and s.find('{') < s.rfind('}') + 1 else ""

def process_with_gpt(system_prompt: str, user_text: str, use_json: bool = False) -> str:
    """Send request to OpenAI-compatible API."""
    try:
        url = f"{OPENAI_BASE_URL}/chat/completions"
        payload = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            "max_tokens": 1000,
            "temperature": 0.5
        }
        if use_json:
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": WEBAPP_URL,
            "X-Title": "Text Processing Bot"
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)
        logger.info(response.text)
        response.raise_for_status()
        result = response.json()
        result = result['choices'][0]['message']['content'].strip()
        return json_block(result)

    except Exception as e:
        logger.error(f"Error in process_with_gpt: {e}", exc_info=True)
        return f"❌ AI processing failed: {str(e)}"

# Add handlers
if application:
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT | filters.Document.ALL, handle_message_or_document))

def transcribe_audio(audio_path: str) -> str:
    try:
        logger.info(f"Transcribing audio from: {audio_path}")
        segments, info = batched_model.transcribe(
            audio_path, multilingual=True, language="ru", batch_size=8,
            vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500)
        )
        logger.info(f"Transcription language: {info.language} (prob: {info.language_probability:.2f})")
        transcription = " ".join(segment.text for segment in segments).strip()
        return transcription
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return ""

def send_to_telegram(chat_id: str, text: str) -> bool:
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logger.error(f"{response.text}")
        return True
    except Exception as e:
        logger.error(f"Error sending to Telegram: {e}", exc_info=True)
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
        logger.info(f"Transcription request from user {user_id}")

        if not audio_data:
            return jsonify({'error': 'No audio data provided'}), 400

        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        chunk_file_path = f"/tmp/{user_id}-{int(datetime.datetime.now().timestamp())}.wav"
        with open(chunk_file_path, 'wb') as chunk_file:
            chunk_file.write(audio_bytes)        
        transcription = transcribe_audio(chunk_file_path)

        if not transcription.strip():
            return jsonify({'status': 'success', 'transcription': ''})

        if chat_id:
            chat_id_str = str(chat_id)
            try:
                ctx = user_context[chat_id_str]
                minutes_left = ctx["duration_minutes"] - (datetime.datetime.now().timestamp() - ctx['start_time']) / 60

                filled_chunk_prompt = f"""
/no_think

Минут до окончания:
{minutes_left}

Текущий отрывок:
{transcription}
"""
                final_response = process_with_gpt(
                    system_prompt=ctx["chunk_system_prompt"],
                    user_text=filled_chunk_prompt,
                    use_json=False
                )

                success = send_to_telegram(chat_id, f"```\n{json.dumps(final_response, indent=2, ensure_ascii=False)}\n```")
            except Exception as e:
                logger.error(f"Error in chained GPT processing: {e}", exc_info=True)
                final_response = transcription
                success = False
        else:
            final_response = transcription
            success = False

        return jsonify({
            'status': 'success',
            'transcription': transcription,
            'enhanced_summary': final_response,
            'sent_to_telegram': success,
            'message': 'Processed successfully' + (' and sent to Telegram' if success else '')
        })
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/webhook', methods=['POST'])
def telegram_webhook():
    try:
        data = request.get_json()
        logger.info(f"Webhook received: {json.dumps(data, indent=2)}")
        if 'message' in data:
            chat_id = data['message']['chat']['id']
            text = data['message'].get('text', '')
            if text.startswith('/start'):
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                payload = {
                    "chat_id": chat_id,
                    "text": "🎤 <b>Welcome!</b>\nOpen the app below:",
                    "parse_mode": "HTML",
                    "reply_markup": {
                        "inline_keyboard": [[{
                            "text": "🎤 Open App",
                            "web_app": {"url": WEBAPP_URL}
                        }]]
                    }
                }
                requests.post(url, json=payload)
        return jsonify({'status': 'ok'})
    except Exception as e:
        logger.error(f"Webhook error: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

def run_flask():
    app.run(host='0.0.0.0', port=7860, debug=False, use_reloader=False)

# Main entry
if __name__ == '__main__':
    import threading
    logger.info("Starting services...")

    # Start Flask server in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("Flask server started on port 7860")

    # Run Telegram bot in the main thread
    if TELEGRAM_BOT_TOKEN:
        logger.info("Telegram bot starting...")
        application.run_polling(drop_pending_updates=True)
    else:
        logger.warning("TELEGRAM_BOT_TOKEN not set. Running Flask only.")
        # Keep main thread alive if no bot
        try:
            while True:
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            pass