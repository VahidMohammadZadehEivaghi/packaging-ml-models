#syntax=docker/dockerfile:1.5

ARG PYTHONVERSION=3.8.10

FROM python:${PYTHONVERSION}-slim as builder 

WORKDIR /app

RUN python -m venv ./venv 

ENV PYTHONDONTWRITEBYTECODE=1 \ 
    PYTHONNUNBUFFERED=1 \
    PATH=/app/venv/bin:$PATH 

RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=bind,source=requirements.txt,target=requirements.txt \
    python -m pip install --upgrade pip & python -m pip install -r requirements.txt


FROM python:${PYTHONVERSION}-slim 

WORKDIR /app 

COPY --from=builder /app/venv /app/venv

ENV PATH=/app/venv/bin:$PATH

COPY . .
CMD ["python", "-m", "uvicorn", "/app/webapp/app:app", "--host", "127.0.0.1", "--port", "8000"]


