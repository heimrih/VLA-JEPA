FROM continuumio/miniconda3:latest

WORKDIR /workspace/VLA-JEPA

COPY environment.cleaned.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml \
    && conda clean -afy

ENV PATH=/opt/conda/envs/vlajepa/bin:$PATH
ENV CONDA_DEFAULT_ENV=vlajepa
ENV PYTHONUNBUFFERED=1

COPY . /workspace/VLA-JEPA

CMD ["bash"]