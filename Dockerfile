# Use the official Python slim image from the Docker Hub
FROM python:3.9-slim


# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    gcc \
    python3-dev \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*


# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY housing_iteration_6_regression1.csv .
COPY test.csv .
COPY housingprediction.py .

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["python3", "housingprediction.py"]