# Base Python image
FROM python:3.10-slim

WORKDIR /app

COPY requirments.txt .
RUN pip install --no-cache-dir -r requirments.txt

# Copy app code
COPY . .

# Run training step
RUN python train_model.py

# Expose the port for Flask
EXPOSE 9555

# Start the app
CMD ["python", "iris_predictor.py"]
