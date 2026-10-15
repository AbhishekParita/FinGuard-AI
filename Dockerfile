FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -m src.data.generate_data --seed 42 --clients 250 --transactions 10000 && python -m src.data.validate_data && python -m src.ml.train_model
EXPOSE 8000
CMD ["uvicorn","src.api.main:app","--host","0.0.0.0","--port","8000"]
