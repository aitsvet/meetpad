import os
import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Configure logging to stderr
import sys
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO,
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Bot is now just a web app launcher - transcription handled by Flask app
logger.info("Bot initialized - transcription handled by web app")

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