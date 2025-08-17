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
from telegram.constants import ChatAction
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

with open('summary.txt', 'r', encoding='utf-8') as file:
    SUMMARY_PROMPT_TEMPLATE = file.read()

with open('minutes.txt', 'r', encoding='utf-8') as file:
    MINUTES_PROMPT_TEMPLATE = file.read()

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
    await update.message.reply_text("""🎤 **Привет!**
MeetPad — твой помощник по проведению встреч.
Пришли мне длительность встречи и список вопросов.""", parse_mode='Markdown')

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

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action=ChatAction.TYPING
    )

    try:
        # First: extract agenda info only once per chat
        agenda_result = process_with_gpt(
            system_prompt=AGENDA_SYSTEM_PROMPT,
            user_text=raw_text
        )
        logger.info(f"Parsed agenda result: {agenda_result}")

        if "questions" not in agenda_result or len(agenda_result["questions"]) == 0 or "duration_minutes" not in agenda_result or agenda_result["duration_minutes"] < 1:
            await update.message.reply_text(f"Не понял, уточните, пожалуйста, длительность встречи в минутах и список вопросов к ней.")
            return

        questions_str = ""
        for (i, q) in enumerate(agenda_result["questions"]):
            questions_str += f"{i+1}. {q}\n"
        user_context[chat_id] = {
            "duration_minutes": agenda_result["duration_minutes"],
            "transcriptions": [],
            "summarized": [],
            "last_summary": 0,
            "in_summary": [],
            "questions": questions_str,
            "summary_system_prompt": SUMMARY_PROMPT_TEMPLATE.format(
                questions=questions_str,
                duration_minutes=agenda_result["duration_minutes"],
                duration_minutes_by_ten=agenda_result["duration_minutes"]/10,
            ),
        }

        sent = await update.message.reply_text(f"""
Понял! У нас будет {agenda_result["duration_minutes"]} минут для обсуждения {len(agenda_result["questions"])} вопросов.
Теперь открой mini-app и нажми на 🎤, чтобы я следил за регламентом встречи.""",
                    parse_mode= "Markdown",
                    reply_markup= {
                        "inline_keyboard": [[{
                            "text": "Открыть mini-app и начать запись 🔴",
                            "web_app": {"url": WEBAPP_URL}
                        }]]
                    })
        user_context[chat_id]["remove_button"] = sent.message_id
    except Exception as e:
        logger.error(f"Error processing message: {e}", exc_info=True)
        await update.message.reply_text("❌ Sorry, an error occurred during processing.")

json_block = lambda s: s[s.find('{'):s.rfind('}')+1] if s.find('{') != -1 and s.rfind('}') != -1 and s.find('{') < s.rfind('}') + 1 else ""

def process_with_gpt(system_prompt: str, user_text: str):
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
            "temperature": 0.5,
            "response_format": {"type": "json_object"}
        }
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": WEBAPP_URL,
            "X-Title": "Text Processing Bot"
        }

        response = requests.post(url, json=payload, headers=headers, timeout=300)
        logger.info(response.text)
        response.raise_for_status()
        result = response.json()
        result = result['choices'][0]['message']['content'].strip()
        return json.loads(json_block(result))

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
        logger.info(f"Transcription result: {transcription}")
        return transcription
    except Exception as e:
        logger.error(f"Transcription error: {e}", exc_info=True)
        return ""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
async def transcribe():
    try:
        data = request.get_json()
        audio_data = data.get('audio_data')
        user_id = data.get('user_id', 'unknown')
        chat_id = str(data.get('chat_id', ''))
        is_finished = data.get('is_finished', False)
        logger.info(f"Transcription request from user {user_id}, meeting {'is' if is_finished else 'not'} finished")

        if not audio_data:
            return jsonify({'error': 'No audio data provided'}), 400

        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        chunk_file_path = f"/tmp/{user_id}-{int(datetime.datetime.now().timestamp())}.wav"
        with open(chunk_file_path, 'wb') as chunk_file:
            chunk_file.write(audio_bytes)        
        transcription = transcribe_audio(chunk_file_path).strip()

        if chat_id:
            ctx = user_context[chat_id]
            if "start_time" not in ctx:
                ctx["start_time"] = int(datetime.datetime.now().timestamp())
                if "remove_button" in ctx:
                    await application.bot.edit_message_reply_markup(
                        chat_id=chat_id,
                        message_id=ctx["remove_button"],
                        reply_markup=None  # Removes the inline keyboard
                    )
                    del ctx["remove_button"]

            if not transcription.strip():
                return jsonify({'transcription': ''})
            ctx["transcriptions"].append(transcription)

            elapsed_time = int(datetime.datetime.now().timestamp()) - ctx["start_time"]
            logger.info(f"in_summary {len(ctx["in_summary"])}, is_finished {is_finished}, " 
                        f"elapsed_time {elapsed_time} - last_summary {ctx["last_summary"]} "
                        f"{'>' if elapsed_time - ctx["last_summary"] > 60 else '<='} 60")

            if is_finished or (len(ctx["in_summary"]) == 0 and elapsed_time - ctx["last_summary"] > 60):
                ctx["in_summary"] = ctx["transcriptions"]
                ctx["transcriptions"] = []
                minutes_left = ctx["duration_minutes"] - elapsed_time // 60
                transcriptions_str = "\n".join(ctx["in_summary"])
                summary_prompt = f"""
/no_think

Минут до окончания:
{minutes_left}

Текущий отрывок:
{transcriptions_str}
"""

                try:
                    if not is_finished:
                        summary = process_with_gpt(
                            system_prompt=ctx["summary_system_prompt"],
                            user_text=summary_prompt
                        )
                        lines = []
                        if "questions" in summary and len(summary["questions"]) > 0:
                            lines.append("*📌 Обсуждалось:*")
                            for q in summary["questions"]:
                                number = q["number"]
                                status = "✅ Завершён" if q["is_resolved"] else "⏳ В процессе"
                                lines.append(f"  • *{number}*: {status}")
                                if q.get("key_findings"):
                                    lines.append(f"    → _{q['key_findings']}_")
                        if "suggestions" in summary:
                            lines.append("*💡 Подсказки:*")
                            for i, suggestion in enumerate(summary["suggestions"], 1):
                                lines.append(f"  • {suggestion}")
                    else: # is_finished
                        summary_prompt = '\n'.join(ctx["summarized"]) + '\n' + '\n'.join(ctx['in_summary'])
                        summary = process_with_gpt(
                            system_prompt=MINUTES_PROMPT_TEMPLATE.format(questions=ctx["questions"]),
                            user_text=summary_prompt
                        )
                        lines = []
                        if "key_points" in summary:
                            lines.append("*📌 Решения:*")
                            for i, point in enumerate(summary["key_points"], 1):
                                lines.append(f"  • {point}")
                        if "action_points" in summary and len(summary["action_points"]) > 0:
                            lines.append("*🛠️ Задачи:*")
                            for i, point in enumerate(summary["action_points"], 1):
                                lines.append(f"  • {point}")

                    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
                    payload = {
                        "chat_id": chat_id,
                        "text": "\n".join(lines).strip(),
                        "parse_mode": "Markdown"
                    }
                    response = requests.post(url, json=payload, timeout=10)
                    response.raise_for_status()

                    ctx["summarized"].append(ctx["in_summary"])                    
                    ctx["last_summary"] = elapsed_time
                    ctx["in_summary"] = []

                    if is_finished:
                        user_context[chat_id] = {
                            "transcriptions": [],
                            "summarized": [],
                            "last_summary": 0,
                            "in_summary": [],
                        }

                    # ctx["remove_button"] = json.loads(response.text)["result"]["message_id"]
                except Exception as e:
                    logger.error(f"Error in summary processing: {e}", exc_info=True)

        return jsonify({'transcription': transcription})
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

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