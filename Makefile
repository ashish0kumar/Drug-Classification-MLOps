install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md

	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/model_results.png)' >> report.md

	cml comment create report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git add . # Add all changed/new files (models, metrics, report)
	git commit -am "Updated with new results and TFLite model" # Update commit message
	git push --force origin HEAD:update

hf-login:
	# Pull from update branch to ensure we have the latest artifacts locally
	git pull origin update
	git switch update
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential

# Push updated files to Hugging Face Space
push-hub:
	echo "Uploading App files to Hugging Face Space..."
	huggingface-cli upload ashish0kumar/Drug-Classificationn ./App --repo-type=space --commit-message="Sync App files"
	echo "Uploading Model files (TFLite, Preprocessors, Label Mapping)..."
	huggingface-cli upload ashish0kumar/Drug-Classificationn ./Model/drug_model_quant.tflite /Model/drug_model_quant.tflite --repo-type=space --commit-message="Sync TFLite Model"
	huggingface-cli upload ashish0kumar/Drug-Classificationn ./Model/cat_imputer.skops /Model/cat_imputer.skops --repo-type=space --commit-message="Sync cat_imputer.skops"
	huggingface-cli upload ashish0kumar/Drug-Classificationn ./Model/encoder.skops /Model/encoder.skops --repo-type=space --commit-message="Sync encoder.skops"
	huggingface-cli upload ashish0kumar/Drug-Classificationn ./Model/num_imputer.skops /Model/num_imputer.skops --repo-type=space --commit-message="Sync num_imputer.skops"
	huggingface-cli upload ashish0kumar/Drug-Classificationn ./Model/scaler.skops /Model/scaler.skops --repo-type=space --commit-message="Sync scaler.skops"
	huggingface-cli upload ashish0kumar/Drug-Classificationn ./Model/label_mapping.skops /Model/label_mapping.skops --repo-type=space --commit-message="Sync label_mapping.skops"
	echo "Uploading Results files..."
	huggingface-cli upload ashish0kumar/Drug-Classificationn ./Results /Metrics --repo-type=space --commit-message="Sync Metrics and Plot"
	echo "Uploads complete."

deploy: hf-login push-hub