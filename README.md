# Hallucinations in Automatic Speech Recognition

This project implements various methods to induce and study hallucinations in Automatic Speech Recognition (ASR) systems. It provides tools for perturbing audio data and analysing the resulting effects on ASR output.
This repository forms part of my dissertation project for the MSc in Speech and Language Processing at the University of Edinburgh, UK.

## Features
- Multiple perturbation methods including:
  - Initial noise injection
  - Speech and noise combination
  - Multi-segment noise injection
  - Reverberation
- Support for various noise types
- Customisable Signal to Noise Ratio (SNR)
- Integration with the TEDLIUM3 dataset

## Installation
1. Clone the repository:
```
git clone https://github.com/zoebrunner/asr-hallucination.git
cd asr-hallucination
```
2. Install the required dependencies:
```
pip install -r requirements.txt
```

## Usage

To run the main script:
```
python hallucination.py [OPTIONS]
```
For a full list of options, run:
```
python hallucination.py --help
```
## License

This project is licensed under the Apache 2.0 License.

Noise files were retrieved from the Music, Speech, and Noise (MuSaN) corpus. All selected recordings were marked as in the Public Domain
on Free Sound (url: https://www.freesound.org/).
