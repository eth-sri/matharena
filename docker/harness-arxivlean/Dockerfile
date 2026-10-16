FROM docker.io/library/node:22-bookworm-slim

ARG ELAN_VERSION=v4.1.2
ARG LEAN_TOOLCHAIN=leanprover/lean4:v4.31.0

ENV DEBIAN_FRONTEND=noninteractive
ENV ELAN_HOME=/elan
ENV VIRTUAL_ENV=/opt/harness-venv
ENV PATH=/opt/harness-venv/bin:/elan/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        python-is-python3 \
        python3 \
        python3-venv \
        ripgrep \
        xz-utils \
        zstd \
    && rm -rf /var/lib/apt/lists/*

RUN curl --fail --location --silent --show-error \
        "https://raw.githubusercontent.com/leanprover/elan/${ELAN_VERSION}/elan-init.sh" \
        -o /tmp/elan-init.sh \
    && sh /tmp/elan-init.sh -y --no-modify-path --default-toolchain "${LEAN_TOOLCHAIN}" \
    && rm /tmp/elan-init.sh \
    && lean --version

COPY python-requirements.txt /opt/harness-python-requirements.txt
RUN python3 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --no-cache-dir --requirement /opt/harness-python-requirements.txt

RUN mkdir -p /opt/mathlib-template /mathlib-cache
COPY lean-toolchain lakefile.lean /opt/mathlib-template/

COPY requirements.toml /etc/codex/requirements.toml
COPY harness-entrypoint /usr/local/bin/harness-entrypoint

RUN mkdir -p /work /run/codex-home \
    && chmod 0755 /work /run/codex-home \
    && chmod 0755 /usr/local/bin/harness-entrypoint \
    && rm -rf /root/.cache /tmp/*

WORKDIR /work

ENTRYPOINT ["/usr/local/bin/harness-entrypoint"]

# Keep Sage's Debian Python separate from the scientific-Python virtualenv.
ARG SAGEMATH_VERSION=9.5-6
ENV SAGE_PYTHON=/usr/bin/python3
RUN apt-get update \
    && apt-get install -y --no-install-recommends "sagemath=${SAGEMATH_VERSION}" \
    && rm -rf /var/lib/apt/lists/* \
    && sage --version \
    && sage -c 'assert ZZ(2026).factor().value() == 2026'

# Preinstall the proof-writing toolchain: evaluation containers cannot install packages.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        latexmk \
        lmodern \
        texlive-fonts-recommended \
        texlive-latex-base \
        texlive-latex-extra \
        texlive-latex-recommended \
        texlive-luatex \
        texlive-pictures \
        texlive-science \
        texlive-xetex \
    && rm -rf /var/lib/apt/lists/*

COPY tex-smoke /opt/harness-tex-smoke
RUN sh /opt/harness-tex-smoke/check.sh
