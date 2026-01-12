# Base image: common for any docker image
FROM python:3.12-slim  

# common for any docker image
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

#Copy over our application (only essential parts for running it) to the docker image
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/
COPY models/ models/

# Step 1: Set the location inside the container
WORKDIR /

# Step 2: Install external dependencies (packages) (--no-cache-dir ensures not saving a copy of downloaded files: Download it, install it, and delete the installer immediately)
#RUN pip install -r requirements.txt --no-cache-dir
#Replace above with below in order to reuse the cache from earlier times the Docker image was built (because we most likely need to rebuild several times due to implementation errors are adding new functionality).
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --no-cache-dir

# Step 3: Install the local project itself (installs current folder (.) as a package) without install dependencies (--no-deps) as these were just installed before
RUN pip install . --no-deps --no-cache-dir

#Set docker entry point
ENTRYPOINT ["python", "-u", "src/cookiecutter_mlops_m6/evaluate.py"]

# In terminal, then run: docker build -f evaluate.dockerfile . -t evaluate:latest
# the -f evaluate.dockerfile . argument indicates which dockerfile we want to run and the -t evaluate:latest is the respective name and tag that we see afterward when running docker images

#After buidling the image, we can check if it succeeded, by running docker images to chack it is there. 

#If this succeeded, we can try to run the docker image: docker run --name experiment1 evaluate:latest