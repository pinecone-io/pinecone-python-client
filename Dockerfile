FROM python:3.8.10-slim

RUN mkdir /code/

COPY ./requirements.txt /code/
RUN pip3 install --upgrade pip && pip3 install -r /code/requirements.txt

COPY ./pinecone /code/pinecone/
COPY setup.py MANIFEST.in README.md LICENSE.txt /code/
RUN pip3 install /code/

RUN rm -r /code/

ENV TINI_VERSION "v0.19.0"
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini
