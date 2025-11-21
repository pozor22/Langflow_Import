import aiohttp
import asyncio
from telebot.async_telebot import AsyncTeleBot

token = "5918045561:AAF8hzcnyXlEeEuHu7r77Yb6Yqp0kve1nu8"
bot = AsyncTeleBot(token=token)


@bot.message_handler(commands=['start'])
async def send_welcome(message):
    user_name = message.from_user.first_name
    welcome_text = f"""
Привет, {user_name}! 🎉

Я ваш телеграм бот. Вот что я умею:

/start - начать работу
/help - помощь
/info - информация о боте

Чем могу помочь?
    """
    await bot.reply_to(message, welcome_text)

# Обработчик команды /help
@bot.message_handler(commands=['help'])
async def send_help(message):
    help_text = """
Доступные команды:

/start - начать работу с ботом
/help - показать эту справку
/info - информация о боте

Вы также можете просто написать мне сообщение, и я отвечу!
    """
    await bot.reply_to(message, help_text)

# Обработчик команды /info
@bot.message_handler(commands=['info'])
async def send_info(message):
    info_text = """
🤖 Информация о боте:

Версия: 1.0
Разработчик: Ваше имя
Используемые технологии: Python, pyTeleBot
    """
    await bot.reply_to(message, info_text)

# Обработчик всех остальных сообщений
@bot.message_handler(func=lambda message: True)
async def echo_all(message):
    await bot.reply_to(message, f"Получил ваше сообщение: '{message.text}' ✨")

# Запуск бота
async def main():
    print("🤖 Бот запущен и готов к работе!")
    print("Остановите бота сочетанием клавиш Ctrl+C")
    await bot.polling(non_stop=True)

try:
    asyncio.run(main())
except KeyboardInterrupt:
    print("Бот остановлен")