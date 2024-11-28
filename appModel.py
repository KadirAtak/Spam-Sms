import logging
from telegram import Update
from telegram.ext import Application, MessageHandler, CallbackContext, filters
import joblib
from telegram.error import BadRequest

# Load your model and vectorizer
model, cv = joblib.load('spam_model.pkl')

# Your bot token
TELEGRAM_TOKEN = '7158434516:AAEeFoQdJoIcrmguTojr1AvtQTzddzW5mYw'

# Setup logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the message handler function
async def handle_message(update: Update, context: CallbackContext):
    message = update.message.text

    message_vector = cv.transform([message]).toarray()
    is_spam = model.predict(message_vector)[0] == 1

    if is_spam:
        user_id = update.message.from_user.id
        chat_id = update.message.chat_id
        try:
            await context.bot.ban_chat_member(chat_id=chat_id, user_id=user_id)
            await update.message.reply_text("Bu mesaj spam olarak tespit edildi ve kullanıcı engellendi.")
        except BadRequest as e:
            if "Can't ban members in private chats" in str(e):
                await update.message.reply_text("Bu mesaj spam olarak tespit edildi, ancak özel sohbetlerde kullanıcıları banlayamam.")
            else:
                raise e

# Define the main function
def main():
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Register handlers
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Start the Bot
    application.run_polling()

if __name__ == '__main__':
    main()
