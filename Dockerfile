# Use Python 3.10.12 slim image as the base
FROM python:3.10.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your handler code
COPY handler.py .
COPY base.py .

# Command to run when the container starts
CMD [ "python", "-u", "handler.py" ]