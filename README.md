# FC-GAGA

This repo provides an implementation of the FC-GAGA algorithm introduced in
https://arxiv.org/abs/2007.15531 and reproduces the experimental
results presented in the paper.

<p align="center">
  <img width="600"  src=./fig/model.png>
</p>

## Citation

If you use FC-GAGA in any context, please cite the following paper:

```
@inproceedings{
  oreshkin2020fcgaga,
  title={{FC-GAGA}: Fully Connected Gated Graph Architecture for Spatio-Temporal Traffic Forecasting},
  author={Boris N. Oreshkin and Arezou Amini and Lucy Coyle and Mark J. Coates},
  booktitle={AAAI},
  year={2021},
}
```

## COLAB based demo

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1HqmvWA-RhcXoCzpgfvUQ4NFLaEEzdeuA)

## Standalone Docker based demo
This workflow can be used to reproduce the FC-GAGA results without relying on the Google Colab environment. All necessary dependencies are captured in ```Dockerfile``` and ```requirements.txt```

Clone this repository
```
mkdir workspace
cd workspace
git clone git@github.com:boreshkinai/fc-gaga.git   
```
Build docker image
```
cd fc-gaga
docker build -f Dockerfile -t fc-gaga:$USER .
```
Start docker container
```
nvidia-docker run -p 8888:8888 -v ~/workspace/fc-gaga:/workspace/fc-gaga -t -d --shm-size="1g" --name fc_gaga_$USER fc-gaga:$USER 
```
Go inside the container and run the main script
```
docker exec -i -t fc_gaga_$USER /bin/bash 
python run.py
```
The script ```run.py``` reproduces all the computations you can see in the Colab notebook.
