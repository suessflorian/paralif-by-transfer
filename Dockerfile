FROM python:3.12.0
WORKDIR /app

RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

COPY Pipfile Pipfile.lock /app/
RUN pip install pipenv

RUN pipenv install
COPY *.py /app/

CMD ["pipenv", "run", "python", "main.py", "rtrainer", "--device", "cpu"]
