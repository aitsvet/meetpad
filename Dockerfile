FROM python:3.11-slim

WORKDIR /app

RUN mkdir -p ./hf_home/
RUN chmod -R 777 ./hf_home/
ENV HF_HOME=/app/hf_home

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose port
EXPOSE 7860

# Run the application
CMD ["python", "app.py"] 