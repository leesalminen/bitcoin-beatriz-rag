FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Update the CMD to include a longer timeout
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "300", "--workers", "4", "app:app"]