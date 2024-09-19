# Hallucinations in Automatic Speech Recognition

This project implements various methods to induce and study hallucinations in Automatic Speech Recognition (ASR) systems. It provides tools for perturbing audio data and analysing the resulting effects on ASR output.
This repository forms part of my dissertation project for the MSc in Speech and Language Processing at the University of Edinburgh, UK.

To run inference on my trained models, download them here: 

https://huggingface.co/collections/zbrunner/hallucinations-in-asr-systems-66e6f9c00ea356700d326d19

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

### Audio Manipulation Options
For a full list of options, run:
```
python hallucination.py --help
```
Alternatively, they are listed here:
- **Basic Functions**:
  - `--plainnoise [DURATION]`: Generate plain noise file (default: 30.0s)
  - `--plainspeech`: Generate plain speech file without perturbation
  - `--speechthennoise [DURATION]`: Create audio with speech followed by noise (default speech: 5.0s)
  - `--noisyspeech`: Generate speech overlaid with noise
  - `--initialnoise [DURATION]`: Add initial noise to speech (default: 3.0s)
  - `--noisethenspeech [DURATION]`: Add noise to beginning and end of speech (default: 3.0s)
  - `--speechandnoise`: Overlay entire speech with noise and replace intervals with noise
  - `--multisegment [NUM_SEGMENTS]`: Join multiple speech segments with noise between (default: 3 segments)

- **Multisegment Options**:
  - `--add_noise`: Enable adding noise over segments in multisegment mode
  - `--variable_snr`: Use variable SNR for multisegment noise (halved during silence)
  - `--silence_only_noise`: Apply noise only to silence segments in multisegment mode

- **Reverberation**:
  - `--reverberation`: Enable reverberation effect
  - `--room_dim`: Set room dimensions for reverberation [length, width, height] (default: [10.0, 7.5, 3.5])
  - `--rt60`: Set reverberation time (RT60) in seconds (default: 0.5)
  - `--use_rand_ism`: Use randomized image source method for reverberation
  - `--max_rand_disp`: Set maximum random displacement for image source method (default: 0.05m)

- **General Options**:
  - `--log, -v`: Enable verbose logging
  - `--outdir`: Specify output directory for created audio files

- **Speech Source**:
  - `--speechsource`: Specify custom speech .wav file
  - `--speakeroverlap`: Use held back training data ('SpeakerOverlap') instead of TEDLIUM3 default test split

- **Noise Customization**:
  - `--snr`: Set Signal to Noise Ratio in dB (default: 15)
  - `--noisetype`: Specify noise type (options: white, crowd, ambient, nature, rain, mechanical)
  - `--noiselength`: Set length of each noise chunk in seconds (default: 3)
  - `--speechlength`: Set interval length between noise chunks in seconds (default: 3)

- **Special Options**:
  - `--reverbonly`: Generate audio with only reverberation applied

## License

This project is licensed under the Apache 2.0 License.

Noise files were retrieved from the Music, Speech, and Noise (MuSaN) corpus. All selected recordings were marked as in the Public Domain
on Free Sound (url: https://www.freesound.org/).
