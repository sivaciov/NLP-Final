# NLP Final Project: Analyzing and Mitigating Dataset Artifacts

This repository contains the implementation for the **NLP Final Project**, focusing on analyzing and mitigating dataset artifacts in pre-trained models. The goal is to identify spurious correlations and biases in datasets and propose strategies to improve model robustness using adversarial and challenge sets.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Methodology](#methodology)
- [Usage](#usage)
  - [Installation](#installation)
  - [Running the Scripts](#running-the-scripts)
- [Example Output](#example-output)
- [Dependencies](#dependencies)
- [License](#license)

## Overview
Dataset artifacts, such as spurious correlations, can lead to models that perform well on benchmarks but fail in real-world applications. This project examines such artifacts in NLP datasets (e.g., SNLI, MultiNLI, SQuAD, HotpotQA) and implements techniques to improve model robustness. The pipeline includes:

1. Training a baseline ELECTRA-small model.
2. Analyzing model performance using adversarial and challenge sets.
3. Extending training with adversarial examples and contrast sets.
4. Evaluating improvements using standard and challenge metrics.

## Features
- **Artifact Analysis:** Identifies and analyzes dataset artifacts using statistical methods and challenge examples.
- **Robust Training:** Extends training data with adversarial and contrast sets to reduce model dependency on spurious correlations.
- **Pre-Trained Models:** Utilizes ELECTRA-small as the base model for experimentation.
- **Comprehensive Evaluation:** Includes metrics such as accuracy, adversarial robustness, and error reduction.

## Methodology
1. **Baseline Training:** Train ELECTRA-small on SNLI, MultiNLI, SQuAD, or HotpotQA.
2. **Artifact Detection:** Use adversarial examples, checklist tests, and statistical analysis to identify failure cases.
3. **Mitigation Strategies:** Design and integrate new training data to address identified artifacts.
4. **Evaluation:** Analyze improvements using both original and adversarial datasets.

## Usage

### Installation
Clone the repository to your local machine:

```bash
git clone https://github.com/sivaciov/NLP-Final.git
cd NLP-Final
```

### Running the Scripts
#### Training the Baseline Model
Run the baseline training script as follows:

```bash
python train_baseline.py --dataset <dataset_name> --epochs <num_epochs> --learning_rate <lr>
```

#### Adversarial Evaluation
Evaluate the model using adversarial datasets:

```bash
python evaluate_adversarial.py --model_path <path_to_model> --adversarial_file <path_to_adversarial_data>
```

#### Training with Enhanced Data
Re-train the model with adversarial and contrast sets:

```bash
python train_with_adversarial.py --dataset <dataset_name> --adversarial_file <path_to_adversarial_data> --epochs <num_epochs> --learning_rate <lr>
```

## Example Output
Sample evaluation output:
```
Baseline Accuracy: 85.4%
Adversarial Set Accuracy: 62.3%
Enhanced Model Accuracy: 78.6%
Error Reduction on Adversarial Set: 27.1%
```

## Dependencies
This implementation uses the following dependencies:
- `numpy`
- `torch`
- `transformers`
- `tqdm`
- `spacy`
- `nltk`

Install the dependencies using:
```bash
pip install numpy torch transformers tqdm spacy nltk
```

Ensure to download the `spacy` language model:
```bash
python -m spacy download en_core_web_sm
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

Feel free to contribute, experiment, and improve the project. Feedback and suggestions are always welcome!
