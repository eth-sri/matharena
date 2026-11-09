# The base is the exact lean-strip runtime recorded by the verified run.
ARG STRIP_IMAGE
FROM public.ecr.aws/docker/library/node:22-bookworm-slim AS node
FROM ${STRIP_IMAGE}
COPY --from=node /usr/local/bin/node /usr/local/bin/node
RUN apt-get update && apt-get install -y --no-install-recommends ripgrep bubblewrap \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/user \
    && ln -s /usr/local /opt/harness-venv \
    && ln -s /opt/elan /elan
COPY toolchains/ /opt/elan/toolchains/
COPY packages/ /opt/leanlean/packages/
COPY tools/ /opt/leanlean/tools/
COPY sources/ /opt/leanlean/sources/
COPY manifest.json /opt/leanlean/manifest.json
COPY pipeline/ /pipeline/
COPY bin/ /usr/local/bin/
COPY requirements.toml /etc/codex/requirements.toml
ENV ELAN_HOME=/elan PYTHONPATH=/pipeline LEAN_NUM_THREADS=8
WORKDIR /work
# Elan otherwise tries to initialize settings on the read-only evaluation root.
RUN printf 'telemetry = false\nversion = "12"\n\n[overrides]\n' > /opt/elan/settings.toml

# New images use host-owned tool boundaries. Historical wrapper tracing is opt-in.
ARG COMMAND_MONITORING=host
RUN if [ "$COMMAND_MONITORING" = wrapper ]; then python -m matharena.monitoring.commands install; fi
RUN python -m leanlean.lean_runtime install

# Cached dependencies need complete Lake hash files before the root is read-only.
RUN python -m leanlean.warm_cache

# Host monitoring needs no collector implementation inside the agent container.
RUN if [ "$COMMAND_MONITORING" = host ]; then \
      rm -rf /pipeline/matharena /pipeline/leanlean/__pycache__; \
      find /pipeline/leanlean -maxdepth 1 -type f \
        ! -name __init__.py ! -name common.py ! -name eval_tools.py \
        ! -name lean_runtime.py ! -name bubblewrap.py -delete; \
    fi
