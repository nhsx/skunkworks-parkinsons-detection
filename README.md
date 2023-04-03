[![MIT License](https://img.shields.io/badge/License-MIT-lightgray.svg)](LICENSE)
![Python Version](https://img.shields.io/badge/Python-3.10.4-blue.svg)
<!-- Add in additional badges as appropriate -->

![Banner of NHS AI Lab Skunkworks ](docs/banner.png)

# NHS AI Lab Skunkworks project: Parkinson's Disease Pathology Prediction using Colourimetry and Machine Learning

> A pilot project for the NHS AI (Artificial Intelligence) Lab Skunkworks team, "Parkinson's Disease Pathology Prediction" (Automatic Sample Decomposition & Detection of Proteinopathy).


## Intended Use

This proof of concept ([TRL 4](https://en.wikipedia.org/wiki/Technology_readiness_level)) is intended to demonstrate the technical validity of applying Machine Learning techniques to a dataset from Imperial College's Brain Bank in order detect Parkinson's disease pathology. It is not intended for deployment in a clinical or non-clinical setting without further development and compliance with the [UK Medical Device Regulations 2002](https://www.legislation.gov.uk/uksi/2002/618/contents/made) where the product qualifies as a medical device.

## Data Protection

This project was subject to a Data Protection Impact Assessment (DPIA), ensuring the protection of the data used in line with the [UK Data Protection Act 2018](https://www.legislation.gov.uk/ukpga/2018/12/contents/enacted) and [UK GDPR](https://ico.org.uk/for-organisations/dp-at-the-end-of-the-transition-period/data-protection-and-the-eu-in-detail/the-uk-gdpr/). No data or trained models are shared in this repository.

## Background

The identification of Parkinson's Disease (PD) from post-mortem brain slices is time consuming for highly trained neuropathologists. Accurate classification and stratification of PD can often take hours per case and is required for the development of research into therapeutics, understanding the progression of PD, and for establishing cause of death.

[Parkinson's UK](https://www.parkinsons.org.uk/) Brain Bank, operating in partnership with [Imperial College London](https://www.imperial.ac.uk/), has produced a dataset containing digitised slices of brains exhibiting various stages of PD pathology; along with control cases, some of which have associated pathology and others which are neurotypical. This dataset is much larger (over 400 cases), more consistent, and of higher quality (all have been stained with the same protocol and imaged within the same laboratory) than has been documented elsewhere in the literature; including those found in a meta-analysis study on detection of neurological disorders containing over 200 papers (Lima et al., 2022).

The project team, consisting of neuroscientists and subject matter experts from [Imperial](https://www.imperial.ac.uk/), [NHS AI Lab Skunkworks](https://transform.england.nhs.uk/ai-lab/), [Parkinson's UK](https://www.parkinsons.org.uk/), and [Polygeist](https://polygei.st) have undertaken a 20 week project through the Home Office's [Accelerated Capability Environment (ACE)](https://www.gov.uk/government/groups/accelerated-capability-environment-ace) to examine the possibility of producing a Proof-of-Concept (PoC) tool to automatically load, and classify those brain scans. The initial focus of the project was to make a tool that could automatically identify a biomarker of PD, the protein 𝛼-synuclein (𝛼-syn); saving time in searching for the protein manually.  A stretch goal of the project was to extent that technology to automatically identify two other proteins (Tau, and β-amyloid). This goal was quickly reached, producing a tool that could detect all three proteins, and effectively measure them.  Making them quickly identifiable for the pathologist, and allowing statistical analyses of individual cases and groups to be conducted. Statistical analysis of the images showed that over 90% of the PD cases could be automatically detected without false alarms, and that the severity and progression of the pathology could be measured.

## Report

A [full technical report](docs/skunkworks-phase-2-technical-report.pdf) including background, model selection, performance metrics and known limitations, and detailing the data pipeline/processes employed will shortly be published on arxiv.org.

## Getting Started

1. Clone this repository (and its submodules) ***N.B. This repo utilises Large File Storage and so requires [git-lfs](https://git-lfs.github.com/) to be available.***
- ```git clone --recurse-submodules -j8 git@github.com:nhsx/skunkworks-parkinsons-detection.git```
Note, if the submodules do not automaticaly download they should be manually downloaded
(this occurs when they are not in the master branch)
- ```git submodule update --init```
2. Create the virtual environment as below

### Environment Setup

The system has been developed on Ubuntu 20.04.4 with the following prerequisites:
- build-essential, wget, git, git-lfs, python3-dev, cmake, libpng-dev, libjpeg-dev, and the CUDA toolkit version 11.7 (i.e. cuda_11.7.0_515.43.04_linux.run).

The system has been tested with python 3.10.4.

To create the virtual environment, run:
```shell
virtualenv venv
. venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-apex.txt

ipython kernel install --name "local-venv" --user
```
Before using the utilities the project must be added to the Python path for each new shell.
```shell
export PYTHONPATH="$PYTHONPATH:$(pwd)"  # Run in the project root folder
```
If the NVIDIA APEX build/install fails for the current environment the CUDA 11.3 library is suggested. Manual installation can then be completed as follows:
```shell
cd ext/apex
CUDA_HOME=/usr/local/cuda-11.3 pip install -v \
  --disable-pip-version-check \
  --no-cache-dir \
  --global-option="--cpp_ext" \
  --global-option="--cuda_ext" \
  ./
```

The codebase consists of a number of libraries for processing pathology samples, and documentation of the development process.  The full processing pipeline requires these samples, however there is a detailed walkthrough in: [A Jupyter Notebook](notebook/PD_Classification_Using_Colourimetry_And_ML.ipynb) (to be viewed via ```jupyter-lab```), and

*Note:* A GPU is required to run these examples. The work has been developed and tested using an NVIDIA RTX 3090 (24GB).

## Codebase Structure

- Data: Sample filenames and conditions, as well as staging information
- docs: Project report
- notebook: End-to-end demonstration of the technique, the work package outputs for exploring the classification of each protein are also available.
- polygeist: Source code
  - CNN: Convolutional Neural Network model, PDNet
  - slidecore: Slide data interface
  - data_faker: Fake data generator (including examples)
  - colour: A colour management library for controlling the calibration of samples
  - virtualscope: A model microscope for modelling the sensor responses of the system used to capture the samples.
- spectral: Calibration information used in the modelling of the microscope and the protein stains.
- ext: External dependencies

## NHS AI Lab Skunkworks
The project is supported by the NHS AI Lab Skunkworks, which exists within the NHS AI Lab at NHSX to support the health and care community to rapidly progress ideas from the conceptual stage to a proof of concept.

Find out more about the [NHS AI Lab Skunkworks](https://www.nhsx.nhs.uk/ai-lab/ai-lab-programmes/skunkworks/).
Join our [Virtual Hub](https://future.nhs.uk/connect.ti/system/text/register) to hear more about future problem-sourcing event opportunities.
Get in touch with the Skunkworks team at [england.aiskunkworks@nhs.net](england.aiskunkworks@nhs.net).

## Licence

Unless stated otherwise, the codebase is released under [the MIT Licence][mit].
This covers both the codebase and any sample code in the documentation.

HTML and Markdown documentation is [© Crown copyright][copyright] and available under the terms
of the [Open Government 3.0][ogl] licence.

The report is licensed under the [Creative Commons Attribution-NoDerivatives 4.0 International licence (CC BY-ND 4.0)][CC-BY-ND4].
Images presented here that are derivative works of biological imagery are used with permission, and are copyright © 2022 Parkinson's UK, and © 2022 Imperial College London, all rights reserved.
Extracts from Zhang and colleagues are used under the MIT License and are © 2022 Cornell University.
The Polygeist logo and title page graphic (which comprises a derivative work used under licence © 2022 Vectorjuice) are © 2022 Polygeist Ltd.
All foreground intellectual property is transferred by Polygeist to the Crown, and all background intellectual property remains that of the respective owners.


[mit]: LICENCE
[copyright]: http://www.nationalarchives.gov.uk/information-management/re-using-public-sector-information/uk-government-licensing-framework/crown-copyright/
[ogl]: http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/
[CC-BY-ND4]: https://creativecommons.org/licenses/by-nd/4.0/legalcode
