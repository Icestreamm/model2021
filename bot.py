import os
import asyncio
import logging
from datetime import datetime
import io
import cv2
import numpy as np
from PIL import Image as PILImage

from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import CommandStart
from aiogram.types import BufferedInputFile
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage

from dotenv import load_dotenv
from ultralytics import YOLO

# ────────────────────────────────────────────────
# 1. الإعدادات الأساسية
# ────────────────────────────────────────────────

load_dotenv()
TOKEN = os.getenv("BOT_TOKEN")
if not TOKEN:
    raise ValueError("BOT_TOKEN غير موجود في المتغيرات البيئية!")

bot = Bot(token=TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

# نموذج بسيط مؤقت (غيّره لاحقًا بنماذجك الحقيقية)
# للتجربة الأولى نستخدم yolov8n فقط، ثم نضيف تحميل نماذجك
model = YOLO("yolov8n.pt")  # هيحمل أوتوماتيك من ultralytics إذا مش موجود

# حالات FSM لجمع بيانات المستخدم (بديل الـ widgets)
class CarAnalysis(StatesGroup):
    waiting_for_photo = State()

# ────────────────────────────────────────────────
# 2. الـ Handlers
# ────────────────────────────────────────────────

@dp.message(CommandStart())
async def start_handler(message: types.Message):
    await message.answer(
        "مرحباً! 👋\n"
        "أنا بوت تحليل أضرار السيارات\n\n"
        "أرسل صورة السيارة وسأبدأ التحليل فوراً\n"
        "ملاحظة: التحليل قد يأخذ 5–30 ثانية"
    )

@dp.message(F.photo)
async def photo_handler(message: types.Message, state: FSMContext):
    try:
        await message.answer("جاري تحميل الصورة وتحليلها... ⏳\n(قد يستغرق 5–30 ثانية حسب السيرفر)")

        # تحميل الصورة كـ bytes
        photo = await message.photo[-1].download(destination=io.BytesIO())
        img_bytes = photo.getvalue()

        # تحويل إلى صيغة OpenCV
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # تشغيل النموذج
        results = model(img_cv, conf=0.30, verbose=False)

        # عدد الاكتشافات
        damage_count = len(results[0].boxes)
        result_text = (
            f"تم اكتشاف {damage_count} منطقة ضرر محتملة\n\n"
            "هذه نتيجة أولية (جاري تطوير النسخة الكاملة مع التقارير والتكاليف)"
        )

        # إنشاء الصورة المعلمة (مع bounding boxes)
        annotated_img = results[0].plot()
        success, buffer = cv2.imencode(".jpg", annotated_img)
        if not success:
            raise Exception("فشل في تحويل الصورة")

        annotated_bytes = buffer.tobytes()

        # إرسال الصورة مع النص
        await message.answer_photo(
            photo=BufferedInputFile(annotated_bytes, filename="damage_analysis.jpg"),
            caption=result_text
        )

        # حفظ حالة (اختياري لاحقًا)
        await state.clear()

    except Exception as e:
        error_msg = f"حدث خطأ أثناء التحليل: {str(e)}"
        logging.error(error_msg)
        await message.answer(error_msg)

# ────────────────────────────────────────────────
# 3. تشغيل البوت
# ────────────────────────────────────────────────

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info("البوت بدأ التشغيل...")
    await dp.start_polling(bot, skip_updates=True)

if __name__ == "__main__":
    asyncio.run(main())
