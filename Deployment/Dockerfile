#Choosing a base image
FROM python:3.13.7-slim

#Copy dependency lists (all the libraries)
COPY Backend/requirements.txt . 

#Install dependencies into container
RUN pip install -r requirements.txt

#Creates a directory in the container (Basically /Deployment on my computer but called /app in the container)
WORKDIR /app

#First dot refers to the current folder Dockerfile is in, 2nd dot refers to the /app folder in the contaienr. Copy will copy over the contents in current folder over to container directory
COPY . .

CMD [ "uvicorn", "Backend.app:app","--host", "0.0.0.0", "--port","8080" ]