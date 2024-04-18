FROM python:latest

EXPOSE 6000

WORKDIR /server
COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
