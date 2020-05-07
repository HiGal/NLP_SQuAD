# NLP SQuAD

## Environment
All development and testing has been done in Conda Python 3 environments on Linux x86-64 and Windows 10  systems, 
specifically Python 3.6.x and 3.7.x.
To create environment with all dependencies in the root directory run
```
conda env create -f environment.yml
conda activate pipeline
```
## Data
To download train and test SQuAD 2.0 data run script `download_dataset.sh`. It loads JSON files into `./data` directory

## Notebooks
1. **EDA.ipynb** - explaratory data analysis on the data. Here we have tried to see how the data look like and what we are dealing with.
2. **Baseline.ipynb** - simple baseline model that was built to make some conclusions about the complexity of the task and about possbile
solution that might work
3. **Simple Albert** - notebook where we have trained just ALBERT to see compare it with next two models.
4. **ALBERT + Autoencoder.ipynb** - notebook where the hypothesis about using Autoencoders with ALBERT was tested
5. **ALBERT + BiLSTMEncoder + ALBERT-SQuAD-OUT.ipynb** - notebook where the hypothesis about using BiLSTMEncoder with ALBERT features was tested.

## Restrospective Reader
In the `retro_mrc` directory you will find the code for retrospective reader and how to run it.
