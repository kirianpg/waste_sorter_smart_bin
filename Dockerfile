

# tensorflow base-images are optimized: lighter than python-buster + pip install tensorflow
FROM tensorflow/tensorflow:2.10.0

COPY requirements.txt /requirements.txt
COPY setup.py /setup.py
COPY project_waste_sorter / project_waste_sorter

RUN pip install -r requirements.txt
RUN pip install --upgrade pip


# NEED TO MODIFY
CMD uvicorn project_waste_sorter.api.api_file:app --host 0.0.0.0 --port $PORT

# $DEL_END
