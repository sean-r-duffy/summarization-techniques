# nlp-final
### Extractive Model

To run this code, simply open the directory **Jacks Extractive Model** and run the `training.py` file. This will output two files: one JSON file to store training and validation results, and another file saving the PyTorch model created. File names can be edited at the bottom of the training loop.

**Note:** If you are using the government reporting dataset, you must ensure the dataset class is updated due to a mismatch in keys (`article` vs. `report` and `abstract` vs. `summary`).

The testing file will run a loop over all the models in the model list provided at the top and will output a JSON file containing the ROUGE scores for each model and each dataset.

