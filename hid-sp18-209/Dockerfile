#Base image
FROM python:3-onbuild

RUN apt-get update
RUN apt-get install -q -y net-tools

# Install pip, Python and matplotlib required libraries
RUN apt-get install -y python python-dev python-pip \
    libxft-dev libfreetype6 libfreetype6-dev
COPY . /app
WORKDIR /app


#CMD ["make", "run"]

# Where API server lives
#WORKDIR /app/

# Install dependencies
COPY requirements.txt /app/
RUN pip install -r ./requirements.txt
EXPOSE 8080
CMD ["make", "run"]

# API server
#COPY api.py /app/
#EXPOSE 7777
#ENTRYPOINT python ./api.py
