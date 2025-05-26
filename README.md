# CrowdGleason

![Approach](Dataset.pdf)
This repo contains associated information with the publication "The CrowdGleason dataset: Learning the Gleason grade from crowds and experts". We include the related citation, the dataset access at Zenodo, and the code to reproduce the experiments.


## Citation 

```
@article{lopez2024crowdgleason,
  title={The CrowdGleason dataset: Learning the Gleason grade from crowds and experts},
  author={L{\'o}pez-P{\'e}rez, Miguel and Morquecho, Alba and Schmidt, Arne and P{\'e}rez-Bueno, Fernando and Mart{\'\i}n-Castro, Aurelio and Mateos, Javier and Molina, Rafael},
  journal={Computer Methods and Programs in Biomedicine},
  volume={257},
  pages={108472},
  year={2024},
  publisher={Elsevier}
}
```

## Data

Dataset publicly available at: https://zenodo.org/records/14178894

## Code

The code is included in the folder code/

### Feature extraction

- code/feature_extraction/ contains the code to train a CNN Prostate clasiffier in SICAP and then extract features from crowdgleason: 1. run train_feat_extractor.py to train the classifier; 2. run predict_features.py to extract features from SICAP and CrowdGleason.

- code/classification/

- code/ablation_study/


