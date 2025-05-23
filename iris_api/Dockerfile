# Base Python image
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Run training step
RUN python train.py

# Expose the port for Flask
EXPOSE 9555

# Start the app
CMD ["python", "app.py"]