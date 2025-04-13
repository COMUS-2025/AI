FROM python:3.9

# Java 17 설치
RUN apt-get update && apt-get install -y openjdk-17-jdk

# JAVA_HOME 환경 변수 설정
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

WORKDIR /app
COPY . /app

# SSL 인증서 복사
COPY server.crt /app/server.crt
COPY server.key /app/server.key

# 필요한 파이썬 라이브러리 설치
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]