# FC-GAGA

This repo provides an implementation of the FC-GAGA algorithm introduced in
https://arxiv.org/abs/2007.15531 and enables reproducing the experimental
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

## Standalone docker based

mkdir workspace

cd workspace

git clone git@github.com:boreshkinai/fc-gaga.git   

cd fc-gaga

docker build -f Dockerfile -t fc-gaga:$USER .

nvidia-docker run -p 8888:8888 -v ~/workspace/fc-gaga/logs:/workspace/fc-gaga/logs -v $/workspace/fc-gaga/data:/workspace/fc-gaga/data -t -d --shm-size="1g" --name fc_gaga_$USER fc-gaga:$USER 
