FROM nvcr.io/nvidia/pytorch:23.04-py3

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND="noninteractive"

# Setup SSH with secure root login
RUN apt update && apt install -y openssh-server netcat \
 && mkdir /var/run/sshd \
 && echo 'root:password' | chpasswd \
 && sed -i 's/\#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]