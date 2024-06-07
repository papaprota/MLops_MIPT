import logging

from aiogram import Bot, Dispatcher, executor, types

from summarization.scripts.infer import infer


logging.basicConfig(
    format="%(levelname)s: %(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
    level=logging.INFO,
)

# ============ !!! Секретный токен !!! ===============
APP_TOKEN = "7437247677:AAE9shT7N44dqdFsshRqJ--_7oIuDI4eyuo"
# ====================================================

bot = Bot(token=APP_TOKEN)
dp = Dispatcher(bot)



@dp.message_handler(commands="all")
async def all_tasks(payload: types.Message):
    await payload.reply(f"Hi")


@dp.message_handler(commands="add")
async def add_task(payload: types.Message):
    text = payload.get_args().strip()
    # new_task = pd.DataFrame({"text": [text], "status": ["active"]})
    # updated_tasks = pd.concat([get_todo_data(), new_task], ignore_index=True, axis=0)

    # ============ Сохраняем ============
    # updated_tasks.to_csv(PATH_TO_TODO_TABLE, index=False)
    # ===================================

    # logging.info(f"Добавил в таблицу задачу - {text}")
    await payload.reply(f"Добавил задачу: *{infer(text=text)}*", parse_mode="Markdown")



if __name__ == "__main__":
    executor.start_polling(dp)