#################### PACKAGE ACTIONS ###################
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
	@mkdir -p models/.lewagon/waste_sorter_smart_bin/data/

# Create relevant subfolders
	@mkdir -p models/.lewagon/waste_sorter_smart_bin/data/processed
	@mkdir -p models/.lewagon/waste_sorter_smart_bin/data/raw

# Create the training_outputs folder
	@mkdir -p models/.lewagon/waste_sorter_smart_bin/training_outputs

# Create relevant subfolders
	@mkdir -p models/.lewagon/waste_sorter_smart_bin/training_outputs/metrics
	@mkdir -p models/.lewagon/waste_sorter_smart_bin/training_outputs/models
	@mkdir -p models/.lewagon/waste_sorter_smart_bin/training_outputs/params

	@echo 'Local setup done with success !'


run_preprocess:
	python -c 'from project_waste_sorter.interface.main import preprocess_vgg16; preprocess_vgg16()'

run_train:
	python -c 'from project_waste_sorter.interface.main import train; train()'

run_pred:
	python -c 'from project_waste_sorter.interface.main import pred; pred()'

run_evaluate:
	python -c 'from project_waste_sorter.interface.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate

run_workflow:
	PREFECT__LOGGING__LEVEL=${PREFECT_LOG_LEVEL} python -m waste_sorter_smart_bin.interface.workflow

run_api:
	uvicorn project_waste_sorter.api.api_file:app --reload

run_streamlit:
	streamlit run project_waste_sorter/frontend/app/app.py

gcp_connect:
	@gcloud auth login
	@gcloud config set project waste-sorter-smart-bin
# To add you account do this :
# gcloud projects add-iam-policy-binding waste-sorter-smart-bin --member="serviceAccount:SERVICE_ACCOUNT_EMAIL" --role="roles/owner"
# SERVICE_ACCOUNT_EMAIL = le-wagon-training@wagon-bootcamp-414210.iam.gserviceaccount.com
# gcloud projects add-iam-policy-binding waste-sorter-smart-bin --member="serviceAccount:le-wagon-training@wagon-bootcamp-414210.iam.gserviceaccount.com" --role="roles/owner"

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
