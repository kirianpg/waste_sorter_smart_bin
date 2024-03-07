reinstall_package:
	@pip uninstall -y project_waste_sorter || :
	@pip install -e .

download_kaggle_data:
	@mkdir -p data/raw_data/kaggle_data
	@curl -L 'https://www.dropbox.com/scl/fi/zp48rvm6atk66xyunofms/archive.zip?rlkey=h1gftcpgedo1qkstdgy7mw7ob&dl=0' -o data/raw_data/kaggle_data.zip
	@unzip -o data/raw_data/kaggle_data.zip -d data/raw_data/kaggle_data/
	@rm data/raw_data/kaggle_data.zip

download_taco_data:
	@mkdir -p data/raw_data/taco_data
	@curl -L 'https://zenodo.org/records/3587843/files/TACO.zip?download=1' -o data/raw_data/taco_data.zip
	@unzip -o data/raw_data/taco_data.zip -d data/raw_data/taco_data/
	@rm data/raw_data/taco_data.zip


local_setup:
# Create the data folder for preprocessing
	@mkdir -p data/
	@mkdir -p data/raw_data
	@mkdir -p data/processed

# Create the data folder for modeling
	@mkdir -p ~/.lewagon/waste_sorter_smart_bin/data/

# Create relevant subfolders
	@mkdir -p ~/.lewagon/waste_sorter_smart_bin/data/processed
	@mkdir -p ~/.lewagon/waste_sorter_smart_bin/data/raw

# Create the training_outputs folder
	@mkdir -p ~/.lewagon/waste_sorter_smart_bin/training_outputs

# Create relevant subfolders
	@mkdir -p ~/.lewagon/waste_sorter_smart_bin/training_outputs/metrics
	@mkdir -p ~/.lewagon/waste_sorter_smart_bin/training_outputs/models
	@mkdir -p ~/.lewagon/waste_sorter_smart_bin/training_outputs/params

	@tree
	@echo 'Local setup done with success !'


################### DATA SOURCES ACTIONS ################

# Data sources: targets for monthly data imports
ML_DIR=~/.lewagon/waste_sorter_smart_bin


reset_local_files:
	rm -rf ${ML_DIR}
	mkdir -p ~/.lewagon/waste_sorter_smart_bin/data/
	mkdir ~/.lewagon/waste_sorter_smart_bin/data/raw
	mkdir ~/.lewagon/waste_sorter_smart_bin/data/processed
	mkdir ~/.lewagon/waste_sorter_smart_bin/training_outputs
	mkdir ~/.lewagon/waste_sorter_smart_bin/training_outputs/metrics
	mkdir ~/.lewagon/waste_sorter_smart_bin/training_outputs/models
	mkdir ~/.lewagon/waste_sorter_smart_bin/training_outputs/params
