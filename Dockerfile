FROM python:3.9

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
COPY ./model.py /code/model.py
COPY ./app.py /code/app.py
COPY ./checkpoints /code/checkpoints

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

CMD ["python", "app.py"] 