DEFAULT_ENV_NAME=qluvio
env-snapshot:
	conda list --explicit > conda-spec-file.txt
	pip list --format freeze > pip-requirements.txt
env-setup: env-setup/$(DEFAULT_ENV_NAME)
env-setup/% :
	conda install -y -m --name $* --file conda-spec-file.txt python=3.6
	(\
		source activate $*;\
	  pip install -r pip-requirements.txt;\
  )
