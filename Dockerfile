FROM nvcr.io/nvidia/pytorch:22.10-py3

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND="noninteractive"

# Setup SSH with secure root login
RUN apt update && apt install -y openssh-server netcat \
 && mkdir /var/run/sshd \
 && echo 'root:password' | chpasswd \
 && sed -i 's/\#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Create splade environment
RUN mkdir /tmp/stage && \
    git clone https://github.com/sdadas/splade /tmp/stage/splade && \
    cd /tmp/stage/splade && \
    conda env create -f conda_splade_env.yml

RUN conda init bash
RUN echo "conda activate splade" >> ~/.bashrc

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]