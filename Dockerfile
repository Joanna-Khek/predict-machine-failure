FROM python:3.8.16
COPY requirements.txt .
RUN pip install -r requirements.txt && rm requirements.txt
EXPOSE 80
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]