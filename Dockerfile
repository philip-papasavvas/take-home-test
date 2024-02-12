# parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Run task_submission.py when the container launches
CMD ["python", "./task_submission.py"]