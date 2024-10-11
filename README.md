**File Structure**
  - _FedLearning_FedAvg.py_ is the server side of the file that aggregates the weights of models from clients using FedAvg strategy
  - _Client_FedAvg.py_ is the file on the client side that trains the model and updates the weights and send them to the server
  - _FedLearning_FedNova.py_ is the server side of the file that aggregates the weights using FedNova strategy

**Note:** To tweak the code for un-weighted FL approach, the client file can be edited by changing the fit function and return instead of len(trainloader) pass 1. This will treat the weights of each client with equal importance. 

**Related Work**
  - "Accuracy vs Privacy: A Federated Learning Approach for LUS Patterns Classification" was presented at UFFC-JS 2024 Taipei
