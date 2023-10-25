Based on demo from https://github.com/Project-MONAI/tutorials/blob/main/automl/DiNTS/search_dints.py and https://github.com/StrongResearch/isc-demos/tree/adam-monai/monai_pancreas_dints

To Run:
Edit the path variables in .isc as well as in configs/search.yaml. In configs/search.yaml the number of epochs has also been reduced for testing, as have nnodes in the .isc.

Status:
The current functionality of the model is that batches are checkpointing and resuming for training with decreasing loss values, but there are issues with loading data in the eval_search funtion. Debug statements have been added to try solve the issue.
