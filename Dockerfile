# Use official Python 3.12 slim image
FROM python:3.12-slim

# Metadata
LABEL authors="Nitesh Saini"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /ml-api

# Install OS-level dependencies
RUN apt-get update && apt-get install -y gcc

# Install Python dependencies
COPY requirement.txt .
RUN pip install --upgrade pip && pip install -r requirement.txt

# Copy all project files into the container
COPY . .


RUN python -m nltk.downloader stopwords wordnet punkt omw-1.4 punkt_tab


# Expose FastAPI port
EXPOSE 8000

# Command to run your app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

