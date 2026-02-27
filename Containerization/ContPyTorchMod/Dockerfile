FROM python:3.10-slim

# set working directory
WORKDIR /ContPyTorchMod

# Install system deps
# Updates the package list and installs libgl1 and libglib2.0-0 (required for OpenCV or image processing)
# Deletes th temporary installation files to keep the image lightweight
RUN apt-get update && apt-get install -y --no-install-recommends \
libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Copies local requirements list into container and install libraries
# By copying only the requirements file first (before the rest of the code), Docker can cache this step
# If you change your code but not your requirements, Docker won't re-install your packages next time you build
# --no-cache-dir: Tells pip not to save the downloaded files, saving more disk space
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies your trained model weights (model.pt) and FastAPI code (app.py) into the /
COPY . .

# the application intends to listen on port 8000. 
# It doesn't actually "open" the port to the public - you do that when you run the container
EXPOSE 8000

# Uvicorn: server that runs FastAPI
# --host 0.0.0.0: tells teh server to listen on all network interfaces inside the container, allowing external traffic to reach it
#app:app - first app is the file name app.py, the second is FastAPI variable inside it
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]

# Didn't finish the docker build steps.