# Gunakan python yang stabil dan ringan
FROM python:3.12-slim

# Set working directory di dalam container
WORKDIR /app

# Copy requirements dulu biar build-nya cepet (caching)
COPY requirements.txt .

# Install dependencies & gunicorn (server production)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

# Copy semua file project ke dalam container
COPY . .

# Buat folder uploads jika belum ada
RUN mkdir -p uploads && chmod 777 uploads

# Port standar Hugging Face Spaces
EXPOSE 7860

# Jalankan Flask pake Gunicorn (Standard Industri)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]