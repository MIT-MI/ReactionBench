FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    ca-certificates \
    ffmpeg \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libsndfile1 \
    libstdc++6 \
    libxext6 \
    libxrender1 \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Path (inside the container) of the mounted host conda env that has `libreface` installed.
# Override at build time: --build-arg LIBREFACE_ENV=/opt/conda/envs/libreface
ARG LIBREFACE_ENV=/opt/conda/envs/libreface
ENV LIBREFACE_ENV=${LIBREFACE_ENV}
ENV CONDA_DEFAULT_ENV=libreface
ENV PATH=${LIBREFACE_ENV}/bin:${PATH}
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONNOUSERSITE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /workspace

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/docker-entrypoint.sh"]
CMD ["python", "--version"]
