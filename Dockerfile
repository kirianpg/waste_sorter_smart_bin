
####### 👇 OPTIMIZED SOLUTION (x86)👇 #######

# tensorflow base-images are optimized: lighter than python-buster + pip install tensorflow
FROM tensorflow/tensorflow:2.10.0

COPY requirements.txt /requirements.txt
COPY setup.py /setup.py

# OR for apple silicon, use this base image instead
# FROM armswdev/tensorflow-arm-neoverse:r22.09-tf-2.10.0-eigen

#WORKDIR /prod

# We strip the requirements from useless packages like `ipykernel`, `matplotlib` etc...
#COPY requirements_prod.txt requirements.txt

COPY project_waste_sorter project_waste_sorter
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


COPY Makefile Makefile
RUN make reset_local_files
RUN make reinstall_package

# NEED TO MODIFY
CMD uvicorn project_waste_sorter.api.api_file:app --host 0.0.0.0 --port $PORT

# $DEL_END
