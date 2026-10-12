FROM mongo:6.0
RUN apt-get update && apt-get install -y --no-install-recommends python3 iproute2 && rm -rf /var/lib/apt/lists/*
COPY common.py run_db.py /app/
ENTRYPOINT []
CMD ["python3", "/app/run_db.py"]
