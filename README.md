# Parameterized Learnable Filters (PLF-Net)

## Abstract
Real-time radio-frequency (RF) sensing applications, such as waveform classification and human activity recognition (HAR), are hindered by conventional two-stage approaches that rely on time-frequency (TF) transforms followed by machine learning. This paper proposes a novel complex-valued neural network architecture that directly classifies raw IQ radar data using parameterized learnable filters (PLFs) as the first layer. Four structured PLFs—Sinc, Gaussian, Gammatone, and Ricker—are introduced to extract frequency-domain features efficiently and interpretably. Evaluated on both experimental and synthetic datasets for signal and modulation recognition, the PLF-based models achieve up to 47% higher accuracy than standard 1D CNNs, and 7% over real-valued filter CNNs, while reducing latency by 75% compared to spectrogram-based 2D CNNs. The results highlight the potential of structured PLFs for accurate, interpretable, and real-time RF sensing.

---

## Setup
1. **Create a Conda environment from the `.yml` file** in this repository:
   ```bash
   conda env create -f environment.yml
   ```
2. **Check the TensorFlow version** (we recommend TensorFlow 2.10). By default, conda may install the latest version:
   ```bash
   python -c "import tensorflow as tf; print(tf.__version__)"
   ```
3. **If TensorFlow is not 2.10**, install it explicitly:
   ```bash
   pip install tensorflow==2.10
   ```
   or
   ```bash
   conda install tensorflow=2.10
   ```
4. **Check the numpy version**, if numpy version is not 1.24.2, install it explicitly:
   ```bash
   pip install numpy==1.24.2
   ```
   or
   ```bash
   conda install numpy==1.24.2
   ```
---

## Usage
- **Main training script**: `train_plfnets.py`
  - Modify it to point to your own dataset if needed.
  - The model expects input data shaped as: **(length, channels, 1)**
    - `length` = 1D timeseries length
    - `channels` = 2 (for real and imaginary parts)
- **Complex Valued Layers**: All custom complex-valued layers are stored in the `complex_valued_layers` folder.

1. Activate the environment:
   ```bash
   conda activate plfnets_env
   ```
2. Run training:
   ```bash
   python train_plfnets.py
   ```

---

## Citation
If you find this work useful, please cite our paper:

@ARTICLE{CV-SincNet,
  author={Biswas, Sabyasachi and Ayna, Cemre Omer and Gurbuz, Sevgi Z. and Gurbuz, Ali C.},
  journal={IEEE Transactions on Radar Systems}, 
  title={CV-SincNet: Learning Complex Sinc Filters From Raw Radar Data for Computationally Efficient Human Motion Recognition}, 
  year={2023},
  volume={1},
  number={},
  pages={493-504},
  keywords={Band-pass filters;Radio frequency;Human activity recognition;Radar imaging;Assistive technologies;Sign language;Radar;RF sensing;FMCW;micro-Doppler signature;CV-SincNet;human activity recognition;ASL},
  doi={10.1109/TRS.2023.3310894}}

## Model Architecture

![Model Architecture](images/sinc_block_updated.png)

*Figure 1: The PLF block used as the intitial layer of the architecture.*

![CV-Sinc Block](images/block_dia.jpg)

*Figure 2: Flow diagram of the PLFNets architecture.*

![Filter Definition](images/filters.png)

*Figure 3: Four different filters and their representation in the time and frequency domain.*

## Interpretability of PLFNets

![filter distribution](images/filter_distribution.png)

*Figure 4: Filter and weight distribution in frequency domain of the learned complex Gaussian filters.*

![Filter Projection](images/filter_md.png)

*Figure 5: Four most important filters on a μ-D spectrogram.*

## Results/Performance comparison

| **Layer**     | **Blocks**  | **CVCNN-1D** | **CNN-1D** | **RV-PLF** | **PLFNet** |
|---------------|-------------|--------------|------------|------------|------------|
| PLF           | (256, 251)  | -            | -          | 1024       | 1536       |
| Conv_input    | (256, 251)  | 130,048      | 129,280    | -          | -          |
| CB 1          | (64, 5)     | 164,480      | 82,240     | 82,240     | 164,480    |
| CB 2          | (64, 5)     | 41,600       | 20,800     | 20,800     | 41,600     |
| CB 3          | (64, 5)     | 41,600       | 20,800     | 20,800     | 41,600     |
| CB 4          | (64, 5)     | 41,600       | 20,800     | 20,800     | 41,600     |
| CB 5          | (64, 5)     | 41,600       | 20,800     | 20,800     | 41,600     |
| CB 6          | (64, 5)     | 41,600       | 20,800     | 20,800     | 41,600     |
| Dense 1       | 256         | 66,048       | 49,408     | 49,408     | 66,048     |
| Softmax       | 100         | 51,400       | 25,700     | 25,700     | 51,400     |
| **Total Parameters** |     | **619,976**   | **390,628**| **262,372**| **491,464**|

*Table 1: Parameter comparison of different models.*

| **Network**     | **Top1** | **Top3** | **Top5** | **Prec.** | **Recall** | **F1**   |
|----------------|---------:|---------:|---------:|----------:|-----------:|---------:|
| CNN-2D         |   63.95  |   81.46  |   89.92  |     66.27 |      62.98 |   63.21  |
| CNN-1D         |   10.82  |   21.79  |   29.16  |     13.65 |       9.98 |   10.08  |
| CVCNN-1D       |   17.63  |   27.94  |   38.89  |     16.81 |      17.76 |   17.15  |
| Sinc           |   56.26  |   75.03  |   81.92  |     61.5  |      56.5  |   55.95  |
| CV-Sinc        |   64.8   |   80.32  |   86.23  |     69.69 |      64.85 |   64.41  |
| Gaussian       |   55.7   |   75.73  |   81.64  |     60.03 |      55.6  |   55.26  |
| **CV-Gauss**   | **65.97**| **83.33**| **89.76**| **70.04** |   **65.8** | **65.27**|
| Ricker         |   58.22  |   76.58  |   84.6   |     64.99 |      55.5  |   57.81  |
| CV-Ricker      |   64.14  |   81.01  |   86.07  |     68.31 |      64.25 |   63.37  |
| Gammatone      |   56.75  |   74.05  |   81.33  |     61.88 |      55.75 |   55.22  |
| CV-Gamma       |   63.71  |   79.74  |   86.29  |     67.76 |      63.6  |   63.53  |

*Table 2: Performance of various models on 100 class ASL recognition using radar signals.*

| **Network**     | **Testing Accuracy** | **Precision** | **Recall** | **F1 Score** |
|----------------|----------------------:|--------------:|-----------:|-------------:|
| CNN2D          | 95.2                 | 95.56         | 96.19      | 95.00        |
| CNN-1D         | 64.6                 | 66.9          | 66.61      | 63.76        |
| CVCNN-1D       | 82.2                 | 77.1          | 83.6       | 77.53        |
| Sinc           | 84.2                 | 90.81         | 81.05      | 79.51        |
| CV-Sinc        | 95.6                 | 95.48         | 95.64      | 95.52        |
| Gaussian       | 90.78                | 90.55         | 90.67      | 90.65        |
| **CV-Gaussian**| **96.8**             | **96.22**     | **96.52**  | **96.67**    |
| Gammatone      | 83.4                 | 88.42         | 83.32      | 79.86        |
| CV-Gammatone   | 94.2                 | 94.61         | 94.41      | 94.61        |
| Ricker         | 86.4                 | 88.9          | 87.07      | 86.48        |
| CV-Ricker      | 95.2                 | 95.9          | 95.32      | 95.24        |

*Table 3: Performance of various models on 5 class RF waveform modulation signals.*

![prediction_time](images/prediction_time.png)

*Figure 6: Testing accuracy vs prediction time for different architectures.*


