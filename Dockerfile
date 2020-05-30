FROM python:3.6
MAINTAINER Micah Melling, micahmelling@gmail.com
RUN mkdir /app
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
RUN git clone https://github.com/micahmelling/covid19
ENTRYPOINT ["python3"]
