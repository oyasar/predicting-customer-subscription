# parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install the package
RUN pip install .

RUN ls -al /app

# Create the output directory
RUN mkdir -p /app/output

EXPOSE 8000

# FLASK_APP environment variable
ENV FLASK_APP=/app/src/predicting_customer_subscription/predict.py

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]