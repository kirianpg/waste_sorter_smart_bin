
####### 👇 OPTIMIZED SOLUTION (x86)👇 #######

# tensorflow base-images are optimized: lighter than python-buster + pip install tensorflow
FROM python:3.10.6-slim

# Installing make
RUN apt-get update && apt-get install -y make

COPY requirements.txt /requirements.txt
COPY setup.py /setup.py
COPY project_waste_sorter /project_waste_sorter

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY Makefile Makefile
#COPY scripts/ /scripts/
RUN make reset_local_files
RUN make local_setup
#RUN make reinstall_package

# NEED TO MODIFY
CMD uvicorn project_waste_sorter.api.api_file:app --host 0.0.0.0 --port $PORT

# $DEL_END
