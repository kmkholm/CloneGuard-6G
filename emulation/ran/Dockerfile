FROM gradiant/ueransim:3.2.6
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-numpy tcpdump procps iputils-ping ethtool \
    && rm -rf /var/lib/apt/lists/*
COPY common.py ran_node.py /app/
ENTRYPOINT []
CMD ["python3", "/app/ran_node.py"]
