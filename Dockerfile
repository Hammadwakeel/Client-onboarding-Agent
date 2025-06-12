# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OPENAI_API_KEY="your_openai_api_key_here"
# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 to allow external access to the app
EXPOSE 8080

# Command to run the app using Gunicorn
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:8080", "app:app"]