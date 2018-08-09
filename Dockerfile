FROM tensorflow/tensorflow:latest-gpu-py3

RUN pip install click \
                tqdm

RUN export LC_ALL=C.UTF-8
RUN export LANG=C.UTF-8

# COPY requirementsDocker.txt /notebooks/requirementsDocker.txt
# RUN pip install -r /notebooks/requirementsDocker.txt
