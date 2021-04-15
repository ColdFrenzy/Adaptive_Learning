FROM rayproject/ray-ml:1.2.0


RUN pip install --upgrade pip 
RUN pip install tensorflow==2.4.0 \
	tqdm \
	gym 

RUN sudo git clone https://github.com/ColdFrenzy/Adaptive_Learning.git
WORKDIR Adaptive_Learning

USER root

ENV PATH  "$PATH:/home/ray/anaconda3/bin:/home/ray/anaconda3/bin:/home/ray/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin"






 


