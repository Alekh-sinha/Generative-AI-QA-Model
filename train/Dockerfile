# Use the official Python 3.10.9 image
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-13.py310:latest

# Copy the current directory contents into the container at .
COPY . .

# Set the working directory to /
WORKDIR /

# Install requirements.txt 
RUN pip install --no-cache-dir --upgrade -r /requirements.txt

# Start the FastAPI app on port 7860, the default port expected by Spaces
CMD [ "python", "train.py"]