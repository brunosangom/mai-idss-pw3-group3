## Intelligent Decision Support System for Wildfire Prediction

This project implements an Intelligent Decision Support System (IDSS) to predict and trigger wildfire alerts based on time-series data. The backend is built with PyTorch and is designed to be configurable and extensible for experimenting with different models and data processing pipelines.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.13+
- pip

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/brunosangom/mai-idss-pw3-group3.git
    cd mai-idss-pw3-group3
    ```

2.  **Install the required Python packages:**
    The backend dependencies are listed in `src/backend/requirements.txt`.
    ```bash
    pip install -r src/backend/requirements.txt
    ```

### Running an Experiment

To start the training and evaluation process, run the `main.py` script and point it to a configuration file. A default `config.yaml` is provided.

```bash
python src/backend/main.py --config src/backend/config.yaml
```

- The script will start the experiment based on the settings in the specified config file.
- Training and validation progress will be logged to `src/backend/logs/experiment.log`.
- The best performing model on the validation set will be saved to `src/backend/checkpoints/best_model.pt`.

### Running the Flask API

To start the Flask API server:

```bash
cd src/backend
flask --app app run --debug
```

The API will be available at `http://localhost:5000` by default. The `--debug` flag enables hot reloading when code changes are saved.

#### Available Endpoints:

- `GET /` - Root endpoint with API information
- `GET /api/health` - Health check endpoint
- TODO

## Backend Implementation

The backend is structured into several key components, each responsible for a specific part of the experiment pipeline.

### 1. Experiment Configuration (`src/backend/config.py`)

The `ExperimentConfig` class is responsible for loading experiment parameters from a YAML file (e.g., `config.yaml`). This allows for easy modification of data paths, feature selection, model hyperparameters, and training settings without changing the code.

### 2. Data Loading and Preprocessing (`src/backend/dataset.py`)

The `WildfireDataset` class, which inherits from `torch.utils.data.Dataset`, handles the data pipeline:

- **Loading**: It loads specified features from the `Wildfire_Dataset.csv`.
- **Grouping**: Data is grouped by location (latitude-longitude pairs) to create individual time-series streams.
- **Splitting**: Each stream is split into training, validation, and test sets based on ratios defined in the config.
- **Normalization**:
    - `latitude` and `longitude` are scaled to a [0, 1] range based on the min/max values in the dataset.
    - `datetime` is converted to the day of the year and scaled by 365 to represent seasonality.
    - All other numerical features are standardized based on the mean and standard deviation of the training and validation sets.
- **Windowing**: The `__getitem__` method returns a sliding window of data (`(window_size, num_features)`) and the corresponding wildfire labels.

### 3. Performance Metrics (`src/backend/metrics.py`)

The `MetricsCollection` class uses the `torchmetrics` library to manage, compute, and track model performance. It is configured to calculate:

- **Precision**
- **Recall**
- **F1-Score**

It provides a simple interface to store predictions and targets during an epoch and then compute the aggregated metrics at the end.

### 4. Model Training (`src/backend/trainer.py`)

The `Trainer` class orchestrates the entire experiment:

- **Initialization**: It sets up the datasets, data loaders, model (currently a Transformer), loss function (Binary Cross-Entropy), and optimizer.
- **Training Loop**: It iterates through the training data for a specified number of epochs, computes the loss, and updates the model weights.
- **Validation**: After each training epoch, it evaluates the model on the validation set. If the validation loss improves, it saves a checkpoint of the model.
- **Logging**: It logs all training, validation, and testing metrics to the file specified in the config.
- **Testing**: After training is complete, it loads the best model checkpoint and evaluates its final performance on the test set.

### 5. Main Entry Point (`src/backend/main.py`)

This script serves as the entry point for the application. It parses command-line arguments to get the path to the configuration file, initializes the `ExperimentConfig`, and kicks off the training process by creating and running a `Trainer` instance.