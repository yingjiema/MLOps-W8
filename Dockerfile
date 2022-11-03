FROM python:3.8
COPY requirements.txt  .
RUN  pip3 install -r requirements.txt
COPY main.py .
CMD ["uvicorn", "main:app","--host", "0.0.0.0"]