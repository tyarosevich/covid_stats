FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY requirements.txt /
RUN pip install -r /requirements.txt
# COPY ./ ./

	
# The code to run when container is started:
COPY application.py .
COPY utils.py .
copy dash_utils.py .
COPY assets/ /app/assets/
# COPY static/ /app/static/
COPY static/ /tmp/
EXPOSE 8080
CMD ["python", "./application.py"]

