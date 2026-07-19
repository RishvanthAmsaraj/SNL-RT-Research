# Reproducible, identical on macOS, Windows, and Linux.
#
#   docker build -t kinarm-rt .
#   docker run -p 8501:8501 kinarm-rt
# then open http://localhost:8501
#
# Uses the conda-forge PyMC build so there is no compiler setup on any OS.
FROM condaforge/miniforge3:latest

WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml && conda clean -afy

# Put the env on PATH so CMD runs inside it.
ENV PATH=/opt/conda/envs/kinarm-rt/bin:$PATH

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
