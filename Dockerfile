FROM aiidalab/full_stack:latest

RUN echo "c.ContentsManager.allow_hidden = True" >> /home/jovyan/.jupyter/jupyter_notebook_config.py
