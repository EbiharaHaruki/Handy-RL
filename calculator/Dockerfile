FROM python:3.10.10

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# user setting
ARG username=docker
ARG useruid=1001
ARG usergid=${useruid}

# install package
RUN apt-get -y update && apt-get upgrade -qqy && apt-get -y install \
    bash \
    curl \
    gcc \
    git \
    jq \
    sudo \
    build-essential \
    g++ \
    make \
    cmake \
    libtool \
    texinfo \
    dpkg-dev \
    pkg-config \
    gfortran \
    libopenblas-dev \
    swig \
    zip \
    vim

RUN pip install --upgrade setuptools

# create group and user
RUN groupadd --gid ${usergid} ${username} && \
    useradd -s /bin/bash --uid ${useruid} --gid ${usergid} -m ${username} && \
    echo ${username}' ALL=NOPASSWD: ALL' >> /etc/sudoers

# Import profile for bash
COPY calculator/bash_profile /home/${username}/.bash_profile
RUN chown ${username}:${username} /home/${username}/.bash_profile

# switch to user
USER ${username}
WORKDIR /home/${username}

# install formatter
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --user -r /tmp/requirements.txt
