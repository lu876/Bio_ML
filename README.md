# Bio_project

This is the official repository for the implementation of our bio_project. 

## Getting Started

Follow these steps to set up your environment and replicate the results:

### Step 1: Configure the Environment

This project requires specific libraries including PyTorch, NumPy, tqdm, pandas, and FrEIA. We recommend using Conda to create a virtual environment and install these dependencies. Execute the following commands:

```
conda create --name your_environment_name python=3.9
conda activate your_environment_name
conda install pytorch numpy tqdm pandas
pip install FrEIA
```

Replace `your_environment_name` with your desired environment name.

### Step 2: Download the Dataset

Please download it from the provided [Click Here](https://drive.google.com/file/d/1xrUcrqLRDz2F2e05OHBh5cIlQ3_AMqvb/view?usp=sharing) and place the files in the `data/1D` directory within this repository.


### Step 3: Train the Network

To start the training process, activate your environment and run the training script:

```
conda activate your_environment_name
cd src
python INNencoder.py
```

### Step 4: Test the Results

After training, evaluate the model to test the results:

```
cd src
# Run the following notebook for testing
run INNencoder-test.ipynb
```

Follow the instructions in `INNencoder-test.ipynb` to execute the test procedures.

