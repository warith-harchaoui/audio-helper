#!/bin/zsh


# This is an attempt to automate the process of setting up a new Python project using Poetry.
# DO NOT RUN IT LIKE A SCRIPT, copy and paste the commands one by one in your terminal.

# Enable debug and error checking
set -x  # Print each command before executing it
set -e  # Exit the script immediately on any command failure

# Configurations
PROJECT_NAME="audio-helper"

# Check if pytorch is okay with your python version (Getting Started section of pytorch.org)
PYTHON_VERSION="3.10"
ENV="env4ah"

DEPENDENCIES=(
    torch
    torchaudio
    ffmpeg-python
    tqdm
    soundfile
    scipy
    git+https://github.com/warith-harchaoui/os-helper.git@main
)


DESCRIPTION="Audio Helper is a Python library that provides utility functions for processing audio files. It includes features like loading audio, converting formats, separating audio sources, and splitting and concatenating audio files."
AUTHORS="Warith Harchaoui <warith.harchaoui@gmail.com>"

conda init
source ~/.zshrc

# Conda environment setup (optional, use only if Conda is required for some reason)
if conda info --envs | grep -q "^$ENV"; then
    echo "Environment $ENV already exists, removing it..."
    conda deactivate
    conda deactivate
    conda remove --name $ENV --all -y
fi


echo "Creating environment $ENV..."
conda create -y -n $ENV python=$PYTHON_VERSION
conda activate $ENV
conda install -y pip

# pip install torch torchaudio

# Convert the dependencies string into an array (compatible with zsh/bash)
# Loop through each dependency and add it with poetry

DEP_ARRAY=(${=DEPENDENCIES})
for dep in "${DEP_ARRAY[@]}"; do
    echo "Adding $dep..."
    pip install "$dep"
done

pip freeze > requirements.txt

# replace git commit hash with @main
sed -i '' 's/@[a-f0-9]\{7,40\}/@main/g' requirements.txt

rm -f poetry.lock pyproject.toml

python requirements_to_toml.py \
    --project_name "$PROJECT_NAME" \
    --description "$DESCRIPTION" \
    --authors "$AUTHORS" \
    --python_version "^$PYTHON_VERSION" \
    --requirements_file "requirements.txt" \
    --output_file "pyproject.toml"




# Poetry setup
pip install --upgrade poetry poetry2setup


# Lock and install dependencies
# submit to chatGPT if toml is not working
poetry install

# # Generate setup.py and export requirements.txt
poetry2setup > setup.py
poetry export -f requirements.txt --output requirements.txt --without-hashes

# # replace git commit hash with @main
sed -i '' 's/@[a-f0-9]\{7,40\}/@main/g' requirements.txt

# Create environment.yml for conda users
cat <<EOL > environment.yml
name: $ENV
channels:
  - defaults
dependencies:
  - python=$PYTHON_VERSION
  - pip
  - pip:
      - -r file:requirements.txt
EOL

echo "Project setup completed successfully!"