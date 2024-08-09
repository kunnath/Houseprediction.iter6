# Houseprediction Iteration 6

## Overview

This project involves predicting housing prices using machine learning models. The code and dependencies are encapsulated within a Docker container to ensure a consistent development and runtime environment. The project includes a Python script that performs the prediction based on the provided dataset.

## Project Structure

- `Dockerfile`: Defines the Docker image and the environment setup.
- `requirements.txt`: Lists the Python packages required for the project.
- `housing_iteration_6_regression1.csv`: Dataset used for training the model.
- `test.csv`: Dataset used for testing the model.
- `housingprediction.py`: Python script that loads the datasets, trains the model, and makes predictions.

## Requirements

- Docker: Ensure that Docker is installed on your system. You can follow the installation instructions from the [official Docker website](https://docs.docker.com/get-docker/).

## Setup

1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Build the Docker image**:
    ```bash
    docker build -t houseprediction:iter6 .
    ```

3. **Run the Docker container**:
    ```bash
    docker run -p 8000:8000 houseprediction:iter6
    ```

    This will start the application inside the container and expose port 8000.

## Usage

Once the container is running, the `housingprediction.py` script will execute automatically. This script will:

1. Load the datasets (`housing_iteration_6_regression1.csv` and `test.csv`).
2. Train a regression model using the data from `housing_iteration_6_regression1.csv`.
3. Make predictions and output results.

## Notes

- Ensure that your `requirements.txt` file accurately reflects all necessary dependencies for your project.
- If you make changes to `housingprediction.py` or other files, rebuild the Docker image using `docker build` to incorporate these changes.

## Troubleshooting

- **Error: `ModuleNotFoundError`**: Ensure that all required Python packages are listed in `requirements.txt`.
- **Container not starting**: Check the Docker logs for errors and ensure all system dependencies are correctly installed.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to customize this README to better fit your project's specifics or any additional instructions you might have.