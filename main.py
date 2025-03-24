import os
import logging
import asyncio
import json
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime
from difflib import SequenceMatcher
import math

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes, \
    ConversationHandler

# Настройка логирования
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Константы для конечного автомата
MAIN, INITIAL_QUESTIONS, ASSESSMENT, STATION = range(4)
GENDER, AGE, FAMILY_HISTORY, SMOKING, DIET, PHYSICAL_ACTIVITY = range(6)

# Системный промпт для модели
SYSTEM_PROMPT = """Ты добрый, отзывчивый и дружелюбный помощник, который готов поддержать и помочь собеседнику. Бережно относишься к чувствительным темам, но много знаешь о здоровье людей и онкологических заболеваниях, готов делиться ими с собеседником в трудную минуту.
Твои задачи:
Предоставление персональных рекомендаций по профилактике рака, Индивидуальная оценка риска развития онкологических заболеваний, Адаптация к собеседнику, Рекомендация обратиться к онкологу или медицинскому генетику
"""

# Few-shot примеры для модели
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "text": "Здравствуйте, я боюсь, что у меня может быть рак."
    },
    {
        "role": "assistant",
        "text": "Привет! Меня зовут Тюльпан, я здесь, чтобы поддержать тебя. Страх перед онкологическими заболеваниями очень распространен, и я понимаю твое беспокойство. Давай поговорим об этом. Что именно вызывает у тебя тревогу? Есть ли конкретные симптомы, которые тебя беспокоят?"
    },
    {
        "role": "user",
        "text": "У меня часто болит голова, это может быть опухоль мозга?"
    },
    {
        "role": "assistant",
        "text": "Понимаю твое беспокойство. Головные боли могут быть вызваны множеством причин, большинство из которых не связаны с онкологией. Это может быть напряжение, стресс, недосыпание, обезвоживание или другие факторы. Опухоль мозга - очень редкая причина головных болей. Тем не менее, если головные боли стали интенсивнее, изменили характер или сопровождаются другими симптомами, важно обратиться к врачу для консультации. Когда у тебя начались эти головные боли?"
    },
    {
        "role": "user",
        "text": "Мне 25 лет, у моей бабушки был рак молочной железы. Я очень боюсь."
    },
    {
        "role": "assistant",
        "text": "Спасибо, что поделилась этой информацией. Понимаю твое беспокойство, особенно учитывая семейную историю. Наличие родственников с раком молочной железы действительно является фактором риска, но важно помнить, что это не означает, что у тебя обязательно разовьется это заболевание. В твоем возрасте рекомендуется: 1) Регулярно проводить самообследование молочных желез; 2) Вести здоровый образ жизни; 3) Посетить врача-маммолога для консультации и определения индивидуального плана наблюдения. Возможно, врач порекомендует начать регулярные обследования раньше, чем обычно. Как ты относишься к идее записаться на консультацию?"
    }
]


class TulipBot:
    def __init__(self, telegram_token: str):
        """Инициализация бота Тюльпан с использованием mock API"""
        self.telegram_token = telegram_token
        # URL mock API-сервера для тестирования (например, запущенного локально)
        self.api_url = "http://localhost:5000/mock_api"
        self.user_data = {}  # Хранение контекста беседы с пользователями

        r_subway = requests.get("https://api.hh.ru/metro/1")
        lines = r_subway.json()["lines"]
        self.cached_subway_stations = [station for line in lines for station in line["stations"]] # Станции метро

        r_clinics = requests.get("https://www.mos.ru/aisearch/facility/clinics/v1/search/?type=36&group_size=7777")
        self.cached_clinics = r_clinics.json()["facility"]["data"]

    def _prepare_prompt(self, user_id: int, message: str) -> dict:
        """Подготовка промпта с учетом контекста пользователя"""
        if user_id not in self.user_data:
            self.user_data[user_id] = {
                "history": [],
                "user_info": {},
                "last_interaction": datetime.now()
            }

        # Добавление нового сообщения в историю
        self.user_data[user_id]["history"].append({"role": "user", "text": message})
        self.user_data[user_id]["last_interaction"] = datetime.now()

        # Создание полного промпта
        messages = [{"role": "system", "text": SYSTEM_PROMPT}]

        # Добавление few-shot примеров, если история пуста
        if len(self.user_data[user_id]["history"]) <= 2:
            messages.extend(FEW_SHOT_EXAMPLES)

        # Добавление истории диалога (с ограничением контекста)
        history = self.user_data[user_id]["history"][-10:]
        for msg in history:
            messages.append({"role": msg["role"], "text": msg["text"]})

        prompt = {
            "modelUri": "mock://test-model",
            "completionOptions": {
                "stream": False,
                "temperature": 0.7,
                "maxTokens": "1500"
            },
            "messages": messages
        }

        return prompt
    
    def _dist_polar(self, lat1, lon1, lat2, lon2):
        return math.hypot(
            lat1 * math.cos(lon1) - lat2 * math.cos(lon2),
            lat1 * math.sin(lon1) - lat2 * math.sin(lon2)
        )
    
    def _find_clinics(self, subway_input):
        station = sorted(self.cached_subway_stations,
                         key=lambda x: SequenceMatcher(None, x["name"],subway_input).ratio(),
                         reverse=True)[0]
        print(station)
        nearest_three = sorted(self.cached_clinics,
                               key=lambda x: self._dist_polar(
                                   x["point"]["lat"], x["point"]["lon"],
                                   station["lat"], station["lng"]))
        nearest_three = nearest_three[:3]
        return nearest_three

    async def generate_response(self, user_id: int, message: str) -> str:
        """Генерация ответа с использованием mock API"""
        prompt = self._prepare_prompt(user_id, message)

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(None, lambda: self._model_inference(prompt))

        # Сохранение ответа в историю
        self.user_data[user_id]["history"].append({"role": "assistant", "text": response})

        return response

    def _model_inference(self, prompt: dict) -> str:
        """Выполнение инференса модели через mock API"""
        headers = {"Content-Type": "application/json"}

        try:
            response = requests.post(self.api_url, headers=headers, json=prompt)
            response.raise_for_status()
            result = response.json()
            if "text" in result:
                return result["text"]
            else:
                logger.error(f"Неожиданный формат ответа mock API: {result}")
                return "Извините, произошла ошибка при обработке запроса."
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка при обращении к mock API: {e}")
            return "Извините, в данный момент я не могу обработать ваш запрос. Пожалуйста, попробуйте позже."
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка при разборе ответа mock API: {e}")
            return "Извините, произошла ошибка при обработке ответа от сервера."
        except Exception as e:
            logger.error(f"Непредвиденная ошибка: {e}")
            return "Извините, произошла непредвиденная ошибка."

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        user = update.effective_user
        user_id = user.id

        logger.info(f"Получена команда /start от пользователя {user_id} ({user.first_name})")

        if user_id not in self.user_data:
            self.user_data[user_id] = {
                "history": [],
                "user_info": {},
                "last_interaction": datetime.now()
            }
            logger.info(f"Инициализированы данные для нового пользователя {user_id}")

        try:
            await update.message.reply_text(
                f"Привет, {user.first_name}! Я Тюльпан - бот, который поможет тебе разобраться "
                "с вопросами о здоровье и онкологических заболеваниях. О чем ты хочешь поговорить сегодня?",
                reply_markup=self._get_main_keyboard()
            )
            logger.info(f"Отправлено приветственное сообщение пользователю {user_id}")
        except Exception as e:
            logger.error(f"Ошибка при отправке приветствия: {e}")

        return MAIN

    def _get_main_keyboard(self) -> InlineKeyboardMarkup:
        keyboard = [
            [InlineKeyboardButton("Персональная оценка риска", callback_data="start_assessment")],
            [InlineKeyboardButton("Рекомендации по профилактике", callback_data="prevention")],
            [InlineKeyboardButton("Развеять мои страхи", callback_data="fears")],
            [InlineKeyboardButton("Где пройти обследование", callback_data="checkup")]
        ]
        return InlineKeyboardMarkup(keyboard)

    async def button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Обработчик нажатий на кнопки"""
        query = update.callback_query
        await query.answer()

        user_id = update.effective_user.id
        callback_data = query.data

        if callback_data == "start_assessment":
            await query.edit_message_text(
                "Для персональной оценки мне нужно задать тебе несколько вопросов о твоем здоровье и образе жизни. Начнем?",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Да, давай начнем", callback_data="begin_questions")],
                    [InlineKeyboardButton("Вернуться назад", callback_data="back_to_main")]
                ])
            )
            return MAIN

        elif callback_data == "begin_questions":
            await query.edit_message_text(
                "Укажи, пожалуйста, твой пол:",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Женский", callback_data="gender_female")],
                    [InlineKeyboardButton("Мужской", callback_data="gender_male")]
                ])
            )
            return INITIAL_QUESTIONS

        elif callback_data.startswith("gender_"):
            gender = callback_data.split("_")[1]
            self.user_data[user_id]["user_info"]["gender"] = gender

            await query.edit_message_text(
                "Спасибо! Теперь укажи, пожалуйста, твою возрастную группу:",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("До 30 лет", callback_data="age_under30")],
                    [InlineKeyboardButton("30-45 лет", callback_data="age_30to45")],
                    [InlineKeyboardButton("46-60 лет", callback_data="age_46to60")],
                    [InlineKeyboardButton("Старше 60 лет", callback_data="age_over60")]
                ])
            )
            return INITIAL_QUESTIONS

        elif callback_data.startswith("age_"):
            age_group = callback_data.split("_")[1]
            self.user_data[user_id]["user_info"]["age_group"] = age_group

            await query.edit_message_text(
                "Есть ли у тебя близкие родственники, у которых был диагностирован рак?",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Да", callback_data="family_yes")],
                    [InlineKeyboardButton("Нет", callback_data="family_no")],
                    [InlineKeyboardButton("Не знаю", callback_data="family_unknown")]
                ])
            )
            return INITIAL_QUESTIONS

        elif callback_data.startswith("family_"):
            family_history = callback_data.split("_")[1]
            self.user_data[user_id]["user_info"]["family_history"] = family_history

            await query.edit_message_text(
                "Спасибо за информацию! Подготовлю персональные рекомендации. Дай мне немного времени..."
            )

            user_info = self.user_data[user_id]["user_info"]
            assessment_request = (
                f"На основе следующей информации о пользователе, предоставь персонализированную оценку "
                f"рисков онкологических заболеваний и рекомендации по профилактике:\n"
                f"Пол: {user_info.get('gender', 'не указан')}\n"
                f"Возрастная группа: {user_info.get('age_group', 'не указана')}\n"
                f"Наличие онкологических заболеваний в семье: {user_info.get('family_history', 'не указано')}\n"
                f"Будь тактичным и вселяй уверенность."
            )

            await context.bot.send_chat_action(chat_id=user_id, action="typing")
            assessment = await self.generate_response(user_id, assessment_request)

            await query.message.reply_text(assessment)
            await query.message.reply_text(
                "Что ты хотел(а) бы узнать дальше?",
                reply_markup=self._get_main_keyboard()
            )
            return MAIN

        elif callback_data == "prevention":
            await context.bot.send_chat_action(chat_id=user_id, action="typing")
            prevention_request = "Расскажи о мерах профилактики онкологических заболеваний для большинства людей."
            response = await self.generate_response(user_id, prevention_request)
            await query.edit_message_text(response)
            await query.message.reply_text(
                "Есть ли у тебя еще вопросы?",
                reply_markup=self._get_main_keyboard()
            )
            return MAIN

        elif callback_data == "fears":
            await context.bot.send_chat_action(chat_id=user_id, action="typing")
            fears_request = "Как справиться с канцерофобией и почему не все симптомы указывают на рак?"
            response = await self.generate_response(user_id, fears_request)
            await query.edit_message_text(response)
            await query.message.reply_text(
                "Что еще тебя интересует?",
                reply_markup=self._get_main_keyboard()
            )
            return MAIN

        elif callback_data == "checkup":
            await query.edit_message_text(
                "Регулярные обследования очень важны. Введи свою станцию метро, а мы найдём ближайшие онкологические диспансеры:"
            )
            return STATION

        elif callback_data == "back_to_main":
            await query.edit_message_text(
                "Чем я могу тебе помочь?",
                reply_markup=self._get_main_keyboard()
            )
            return MAIN

        return MAIN

    async def handle_subway(self, update: Update, _context: ContextTypes.DEFAULT_TYPE) -> int:
        """Обработчик станции метро"""
        name = update.message.text
        nearest = self._find_clinics(name)
        for i in nearest:
            await update.message.reply_text(f"{i["name"]}\n\nАдрес: {i["main_address"]}")

        await update.message.reply_text(
            "Что еще ты хотел(а) бы узнать?",
            reply_markup=self._get_main_keyboard()
        )
        return MAIN

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Обработчик текстовых сообщений"""
        user_id = update.effective_user.id
        message_text = update.message.text

        await context.bot.send_chat_action(chat_id=user_id, action="typing")
        response = await self.generate_response(user_id, message_text)
        await update.message.reply_text(response)

        return MAIN

    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Обработчик команды /help"""
        help_text = (
            "Я Тюльпан - твой помощник в вопросах здоровья и онкологии.\n\n"
            "Я могу:\n"
            "• Предоставлять персональные рекомендации\n"
            "• Оценивать риски и рассказывать о профилактике\n"
            "• Поддерживать и развеивать страхи\n\n"
            "Просто напиши свой вопрос или выбери опцию из меню."
        )

        await update.message.reply_text(help_text)

    async def save_data(self, context: Optional[ContextTypes.DEFAULT_TYPE]) -> None:
        """Периодическое сохранение данных пользователей"""
        now = datetime.now()
        users_to_remove = []
        for user_id, data in self.user_data.items():
            if (now - data["last_interaction"]).total_seconds() > 86400:
                users_to_remove.append(user_id)
        for user_id in users_to_remove:
            del self.user_data[user_id]
        logger.info(f"Очищены данные {len(users_to_remove)} неактивных пользователей")

        try:
            with open('user_data.json', 'w', encoding='utf-8') as f:
                serializable_data = {}
                for user_id, data in self.user_data.items():
                    serializable_data[str(user_id)] = {
                        "history": data["history"],
                        "user_info": data["user_info"],
                        "last_interaction": data["last_interaction"].isoformat()
                    }
                json.dump(serializable_data, f, ensure_ascii=False, indent=2)
            logger.info("Данные пользователей успешно сохранены")
        except Exception as e:
            logger.error(f"Ошибка при сохранении данных: {e}")

    def load_data(self) -> None:
        """Загрузка сохраненных данных пользователей"""
        try:
            if os.path.exists('user_data.json'):
                with open('user_data.json', 'r', encoding='utf-8') as f:
                    serialized_data = json.load(f)
                for user_id, data in serialized_data.items():
                    self.user_data[int(user_id)] = {
                        "history": data["history"],
                        "user_info": data["user_info"],
                        "last_interaction": datetime.fromisoformat(data["last_interaction"])
                    }
                logger.info(f"Загружены данные {len(self.user_data)} пользователей")
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}")

    async def _periodic_save(self, interval: int) -> None:
        """Периодическое сохранение данных если job_queue недоступна"""
        while True:
            await asyncio.sleep(interval)
            await self.save_data(None)

    async def run(self):
        """Запуск бота"""
        self.load_data()
        application = Application.builder().token(self.telegram_token).build()

        conv_handler = ConversationHandler(
            entry_points=[CommandHandler("start", self.start)],
            states={
                MAIN: [
                    CallbackQueryHandler(self.button_handler),
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
                ],
                INITIAL_QUESTIONS: [CallbackQueryHandler(self.button_handler)],
                ASSESSMENT: [
                    CallbackQueryHandler(self.button_handler),
                    MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
                ],
                STATION: [MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_subway)]
            },
            fallbacks=[CommandHandler("start", self.start)],
            name="tulip_conversation",
            persistent=False
        )

        application.add_handler(conv_handler)
        application.add_handler(CommandHandler("help", self.help_command))

        job_queue = application.job_queue
        if job_queue is not None:
            job_queue.run_repeating(self.save_data, interval=1800)
        else:
            logger.warning("JobQueue не доступен. Будет использоваться альтернативное решение для сохранения данных.")

        await application.initialize()
        await application.start()
        await application.updater.start_polling()
        logger.info("Бот Тюльпан запущен")

        try:
            if job_queue is None:
                save_task = asyncio.create_task(self._periodic_save(1800))
            stop_signal = asyncio.Future()
            await stop_signal
        except (KeyboardInterrupt, SystemExit):
            logger.info("Получен сигнал остановки")
        finally:
            await self.save_data(None)
            if job_queue is None and 'save_task' in locals() and not save_task.done():
                save_task.cancel()
            if hasattr(application.updater, 'stop'):
                await application.updater.stop()
            await application.stop()
            await application.shutdown()


if __name__ == "__main__":
    TELEGRAM_TOKEN = os.getenv("TG_TOKEN", "INSERT_YOUR_TELEGRAM_TOKEN")

    if not TELEGRAM_TOKEN:
        logger.error("Не указан обязательный параметр TELEGRAM_TOKEN.")
        exit(1)

    bot = TulipBot(TELEGRAM_TOKEN)
    try:
        asyncio.run(bot.run())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Завершение работы бота")
