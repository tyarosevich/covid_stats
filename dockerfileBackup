FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "covid_stats_env", "/bin/bash", "-c"]

# Environment variables for the app.
	
# The code to run when container is started:
COPY application.py .
COPY utils.py .
copy dash_utils.py .
COPY assets/ /app/assets/
COPY static/ /app/static/
EXPOSE 8080
ENTRYPOINT ["conda", "run", "-n", "covid_stats_env", "python", "application.py"]

