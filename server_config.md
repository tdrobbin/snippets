fist install miniconda/anaconda
```
# install miniconda
# choose all default settings so everything after works
echo 'installing miniconda'
curl -O https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
echo 'installation done'

# add conda to path
echo export PATH=$PATH:$HOME/miniconda3/bin >> $HOME/.bashrc

# activate installation
source $HOME/.bashrc
```

then configure jupyterlab to access via https
```
conda install jupyter nb_conda_kernels
```

to set up the server to run remote and hav web access.
(https://jupyter-notebook.readthedocs.io/en/stable/public_server.html)
first make a ssl certificate
```
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /root/ssl/mykey.key -out /root/ssl/mycert.pem
```
then config the server
```
jupyter notebook --generate-config
jupyter notebook password
```

and add this to ~/.jupyter/jupyter_notebook_config.py
```
echo "
c.NotebookApp.certfile = u'/root/ssl/mycert.pem'
c.NotebookApp.keyfile = u'/root/ssl/mykey.key'
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 9999
c.NotebookApp.allow_remote_access = True
c.MultiKernelManager.default_kernel_name = 'base'
c.ContentsManager.allow_hidden = True
" > ~/.jupyter/jupyter_notebook_config.py
```

to run the jab app on the server and optionally detach
```
jupyter lab --allow-root
```
```
nohup jupyter lab --allow-root >/dev/null 2>&1 &
```

now install some linux stuff
```
apt-get update
apt-get install build-essential libpq-dev
apt-get install postgresql postgresql-contrib
#apt-get install postgresql
```

no install python sql stuff
```
pip install dataset
pip install psycopg2
```

to use sql from bash
```
su - postgres
psql
```
