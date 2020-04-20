FROM python:3
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY main.py main.py
COPY input_data.csv input_data.csv
ENTRYPOINT [ "python3", "main.py" ]
