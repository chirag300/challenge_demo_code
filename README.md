# Event-Based Eye Tracking

This repository contains code files and data necessary for event-based eye tracking tasks. 

## Overview

Event-based eye tracking is a critical component in various applications such as augmented reality, virtual reality, and neuroscience research. This repository provides a comprehensive solution for training models to accurately track eye movements using event-based data.

## Contents

- **Code Files:** Contains all the code files required for training and evaluation of eye tracking models.
- **Data Files:** Includes the dataset required for training and testing the eye tracking models.
- **Results:** Stores the results obtained from the trained models.
- **Documentation:** Includes detailed documentation explaining the code structure, model architectures, and usage instructions.

## Important Files

- `train.py`: Main script for training eye tracking models.
- `test.py`: Script for evaluating trained models on test data.
- `model/`: Folder containing implementations of different eye tracking model architectures.
- `requirements.txt`: Lists all the required dependencies for running the code.

## Execution Steps

To start the execution, you need to pull out the `event_data` folder that contains the dataset to execute the code. This step is very important.

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using `pip install -r requirements.txt`.
4. Pull out the `event_data` folder containing the dataset.
5. Run the training script using `python train.py`.
6. Evaluate the trained model on test data using `python test.py`.
7. Explore the results and documentation for further insights.

## Additional Information

For executing the code in Google Colab, you can follow these easy steps:

1. Install Miniconda:

```bash
!wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
!chmod +x Miniconda3-latest-Linux-x86_64.sh
!bash ./Miniconda3-latest-Linux-x86_64.sh -b -f -p /usr/local

import sys
sys.path.append('/usr/local/lib/python3.7/site-packages/')

!conda install -y -c conda-forge python=3.9


2. Clone the repository and navigate to the project directory:

```bash
!git clone https://github.com/chirag300/challenge_demo_code.git
%cd challenge_demo_code/
```

3. Create the conda environment and install dependencies:

```bash
!conda env create -f environment.yml
```

4. Install additional dependencies:

```bash
!pip install mlflow
!pip install h5py
!pip install tonic
!pip install torchvision
```

5. Activate the conda environment and run the training script:

```bash
%%shell
eval "$(conda shell.bash hook)" # copy conda command to shell
conda activate event_eyetracking
python3 train.py --config sliced_baseline.json --num_epochs 100
```

These are easy steps to execute the files in Google Colab.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

For more information, please refer to the [documentation](documentation/README.md).
```
This readme provides a detailed guide on executing the code, including steps for setting up the environment in Google Colab and accessing important files.
