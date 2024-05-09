FROM python:3.9

WORKDIR .

COPY requirements.txt ./
COPY api_titanic.py ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


COPY . .
CMD ["python", "./api_titanic.py"]