FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends iproute2 procps iputils-ping ethtool tcpdump \
    && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir numpy onnxruntime pymongo
COPY common.py phy_auth.py /app/
CMD ["python3", "/app/phy_auth.py"]
