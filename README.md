# سامانه پیش‌بینی احتمال وجود سنگ کیسه صفرا

اپلیکیشن فارسی Streamlit با استفاده از مدل Random Forest آموزش‌دیده و ذخیره‌شده؛
مدل در این پروژه مجدداً آموزش داده نمی‌شود.

## ساختار پروژه

```text
.
├── app.py
├── README.md
├── requirements.txt
├── render.yaml
├── .gitignore
├── .python-version
├── data/
│   └── gallstone.csv
├── models/
│   └── gallstone_model.joblib
├── reports/
│   └── model_metrics.json
```

## نصب و اجرا

مدل با `scikit-learn 1.6.1` ذخیره شده است. برای جلوگیری از ناسازگاری،
از Python 3.11 و نسخه‌های ثبت‌شده در `requirements.txt` استفاده کنید:

```bash
conda create -n gallstone-app python=3.11.18
conda activate gallstone-app
python -m pip install -r requirements.txt
streamlit run app.py
```

## استقرار روی Render

تنظیمات Render در `render.yaml` تعریف شده و نسخهٔ Python از `.python-version`
خوانده می‌شود. برای استقرار:

1. تغییرات را Commit و به GitHub Push کنید.
2. در Render یک Blueprint جدید بسازید و این مخزن را متصل کنید؛ Render تنظیمات
   `render.yaml` را می‌خواند. در صورت ساخت Web Service به‌صورت دستی، ریشهٔ مخزن
   را به‌عنوان Root Directory انتخاب کنید.
3. دستور Build برابر `pip install -r requirements.txt` و دستور Start برابر
   دستور زیر است:

   ```bash
   streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
   ```

استقرار یا Deploy از داخل این پروژه انجام نشده است.

## فرم و پیش‌بینی

فرم شامل ۳۸ ویژگی مدل در چهار بخش است. ورودی‌ها با ترتیب و نام ویژگی‌هایی که
از فایل مدل استخراج می‌شوند به مدل فرستاده می‌شوند. فیلدهای دودویی از
پرتکرارترین مقدار دیتاست و فیلدهای عددی از میانهٔ همان دیتاست مقدار اولیه
می‌گیرند. واحدی که در فایل‌های پروژه مشخص نشده باشد در رابط نمایش داده نمی‌شود.

خروجی شامل کلاس پیش‌بینی‌شده و احتمال‌های هر دو کلاس است:

- کلاس ۰: دارای سنگ کیسه صفرا
- کلاس ۱: بدون سنگ کیسه صفرا

**هشدار:** این سامانه صرفاً آموزشی و پژوهشی است و ابزار تشخیص پزشکی نیست.
نتیجهٔ مدل نباید مبنای تشخیص یا تصمیم درمانی قرار گیرد.
