import logging
import os

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, executor, types

from summarization.scripts.test import infer

load_dotenv()

logging.basicConfig(
    format="%(levelname)s: %(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)
# ============ !!! Секретный токен !!! ===============
APP_TOKEN = os.getenv("APP_TOKEN")
# ====================================================
print(f"tockern  = {APP_TOKEN}")

bot = Bot(token=APP_TOKEN)
dp = Dispatcher(bot)



@dp.message_handler(commands="hi")
async def all_tasks(payload: types.Message):
    await payload.reply(f"Hi")


@dp.message_handler(commands="summarize")
async def add_task(payload: types.Message):
    try:
        text = payload.get_args().strip()
        
        await payload.reply(f"Добавил задачу: *{infer(text=text)}*", parse_mode="Markdown")

    except Exception as e:

        await payload.reply(f"""{e}   
                            Было очень интересно, но я ничего не понял, попробуй еще раз""", parse_mode="Markdown")






if __name__ == "__main__":
    executor.start_polling(dp)