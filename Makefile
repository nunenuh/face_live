export-env:
	conda activate facial & conda env export > environment.yml

run:
	conda activate facial & python run.py

