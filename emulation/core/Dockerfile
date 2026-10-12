FROM gradiant/open5gs:2.7.2
RUN apt-get update && apt-get install -y --no-install-recommends python3 procps iputils-ping ethtool \
    && rm -rf /var/lib/apt/lists/*
COPY gen_configs.py common.py run_core.py /app/
CMD ["python3", "/app/run_core.py"]
