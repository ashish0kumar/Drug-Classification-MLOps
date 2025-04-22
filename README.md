# ML-DevOps: Drug Classification with CI/CD and Hugging Face Deployment

This project demonstrates a basic Machine Learning workflow incorporating DevOps
practices, including Continuous Integration (CI) and Continuous Deployment (CD).
It focuses on training a simple drug classification model, evaluating it, and
deploying it as a web application on Hugging Face Spaces.

## Project Goal

- Data loading and preprocessing.
- Training a machine learning model.
- Evaluating the model.
- Automating the training, evaluation, and deployment process using GitHub
  Actions.
- Deploying the model and a Gradio-based web application to Hugging Face Spaces
  for easy interaction.

## CI/CD Pipelines (GitHub Actions)

The project includes GitHub Actions workflows to automate the CI/CD process.

- **Continuous Integration** (`.github/workflows/ci.yml`):

  - Triggered on pushes and pull requests to the main branch, and manually via
    `workflow_dispatch`.

  - Checks out the code.

  - Sets up CML.

  - Installs dependencies (`make install`).

  - Formats the code (`make format`).

  - Trains the model (`make train`).

  - Evaluates the model (`make eval`) and uses CML to comment the results on the
    pull request.

  - Updates a separate branch (`update`) with the latest results and model files
    (`make update-branch`). This branch is used by the CD pipeline.

- **Continuous Deployment** (`.github/workflows/cd.yml`):

  - Triggered upon completion of the `Continuous Integration` workflow or
    manually via `workflow_dispatch`.

  - Checks out the code.

  - Logs in to Hugging Face Hub using a secret token (`HF`).

  - Pushes the App, Model, and Results directories to the specified Hugging Face
    Space (`ashish0kumar/Drug-Classificationn`) using `huggingface-cli upload`.

## Deployment on Hugging Face Spaces

The project is designed to be deployed as a Hugging Face Space. The `App`
directory contains the necessary files (`drug_app.py`, `requirements.txt`,
`README.md` for Space configuration). The CI/CD pipeline automates the process
of updating the Space with the latest model and application code.

You can access the deployed application at: <br>
`https://huggingface.co/spaces/ashish0kumar/Drug-Classificationn`

## Using the Deployed App

You can interact with the drug classification model through the web interface.
Simply input the patient's features (Age, Sex, Blood Pressure, Cholesterol,
Na_to_K ratio), and the app will provide the predicted drug type.
