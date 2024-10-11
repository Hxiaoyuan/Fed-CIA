# Fed-CIA
Federated causal intervention aggregation model for identifying rare diseases in chest X-ray images


## Datasets

The project utilizes multiple datasets for evaluation in different federated learning contexts, including medical diagnosis and image classification. 
Below are the datasets used along with their download links:

### 1. Chest X-ray Dataset
- **Download Link**: [Chest X-ray Dataset](https//nihcc.app.box.com/v/ChestXray-NIHCC)

### 2. CIFAR-10

- **Download Link**: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz)

### 3. CIFAR-100
- **Download Link**: [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz)



## Installation and Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Fed-CIA.git
   cd Fed-CIA
   python FedAvg.py # Run the FedAvg algorithm
   python FedAvg_intervention.py # Run the FedAvg algorithm with added causal intervention module
   python FedAvg_intervention_ada.py # Run FedAvg algorithm with added adaptive causal intervention module
```

2. Project Structure
Fed-CIA/  
├── dataset/  
│   └── *All Dataset*  
│  
├── Model/  
│   ├── **ClassificationM.py**   *Model architecture for classification tasks*  
│   └── **model_res.py**         *ResNet-based model*  
│  
├── util/  
│   ├── **dataset.py**           *Dataset-related utilities*  
│   ├── **fedavg.py**            *Implementation of the FedAvg algorithm*  
│   ├── **fednh.py**             *Implementation of FedNH (Neural Heterogeneity)*  
│   ├── **imbalance_cifar.py**   *Utilities for handling imbalanced datasets*  
│   ├── **losses.py**            *Custom loss functions*  
│   ├── **sampling.py**          *Data sampling methods*  
│   └── **util.py**              *General utility functions*  
│  
├── **FedAvg.py**                     *FedAvg algorithm*  
└── **FedAvg_intervention.py**     *FedAvg algorithm with added causal intervention module*  
```

