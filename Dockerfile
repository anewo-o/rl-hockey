FROM mambaorg/micromamba:2.5.0

COPY --chown=$MAMBA_USER:$MAMBA_USER conda-lock.yml /tmp/conda-lock.yml

RUN micromamba create -y -n ift702-docker -f /tmp/conda-lock.yml && \
micromamba clean --all --yes

RUN micromamba run -n ift702-docker AutoROM --accept-license
    # --install-dir $CONDA_PREFIX/lib/python3.13/site-packages/ale_py/roms

COPY index.py /tmp/index.py

CMD ["micromamba", "run", "-n", "ift702-docker", "python", "index.py"]