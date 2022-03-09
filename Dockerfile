FROM python:3
WORKDIR /usr/src/app

COPY requirements.txt ./

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8501

COPY . .

CMD [ "streamlit", "run", "./main.py" ]
