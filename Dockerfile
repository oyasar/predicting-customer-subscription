# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install the package
RUN pip install .

RUN ls -al /app

# Create the output directory
RUN mkdir -p /app/output

# Make port 80 available to the world outside this container
EXPOSE 8000

# Set the FLASK_APP environment variable
ENV FLASK_APP=/app/src/predicting_customer_subscription/predict.py

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]