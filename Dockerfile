# Use Python 3.11 slim image
FROM python:3.11-slim

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user

# Set working directory to the user's home directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements file from the subdirectory
COPY --chown=user backend/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend application code from the subdirectory
COPY --chown=user backend/ .

# Fix permissions for /app directory to allow writing
RUN chown -R user:user /app

# Make the start script executable
RUN chmod +x start.sh

# Switch to the "user" user
USER user

# Set environment variables
ENV PORT=7860
ENV PATH="/home/user/.local/bin:$PATH"
ENV PYTHONPATH=/app

# Expose the port
EXPOSE 7860

# Run the start script
CMD ["./start.sh"]
