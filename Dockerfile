# Use an official lightweight Python runtime
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Tell Docker that the app will listen on port 8000
EXPOSE 8000

# Copy the dependency list first to leverage Docker cache
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY main.py .

# Expose the environment variable (default value can be empty or set here)
ENV OLLAMA_BASE_URL=""

# Run the script
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
