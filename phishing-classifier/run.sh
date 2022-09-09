#!/bin/bash
# create a temporary virtual-environment and activate it
echo "CREATING TEMPORARY VENV AND ACTIVATING!"
python3 -m venv piicf-temp
source piicf-temp/bin/activate

# check if we are in a venv
if [[ "$VIRTUAL_ENV" != "" ]]
then
    # install requirements
    echo "VENV ACTIVATED; INSTALLING PIP REQUIREMENTS"
    pip install -r requirements.txt

    # run script
    echo "RUNNING SCRIPT (THIS MAY TAKE SEVERAL MINUTES DEPENDING ON YOUR HARDWARE)"
    python3 random-forest-classifier.py

    # deactivate venv and throw it away
    echo "DEACTIVATING VENV AND TRASHING IT"
    deactivate
    rm -rf ./piicf-temp
else
    echo "Creating and activating the virtual environment failed. Do you have the venv module?"
fi