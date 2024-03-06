download_kaggle_data:
	@mkdir -p raw_data/kaggle_data
	@curl -L 'https://www.dropbox.com/scl/fi/zp48rvm6atk66xyunofms/archive.zip?rlkey=h1gftcpgedo1qkstdgy7mw7ob&dl=0' -o raw_data/kaggle_data.zip
	@unzip -o raw_data/kaggle_data.zip -d raw_data/kaggle_data/
	@rm raw_data/kaggle_data.zip

download_taco_data:
	@mkdir -p raw_data/taco_data
	@curl -L 'https://zenodo.org/records/3587843/files/TACO.zip?download=1' -o raw_data/taco_data.zip
	@unzip -o raw_data/taco_data.zip -d raw_data/taco_data/
	@rm raw_data/taco_data.zip
