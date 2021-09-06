# Physical Property Learner and Dynamic Predictor
 This repo contains the physical property learner and dynamic predictor of Compositional Physics Learner in ComPhy [paper](https://openreview.net/pdf?id=wgQoeAdyk11)

### Data preparation
- Download training and validation object tracks of [target videos](https://drive.google.com/file/d/1hr_x85_FqKp17z7ZdNOoKrQK3EthbkiQ/view?usp=sharing) and [reference videos](https://drive.google.com/file/d/1F16qhzsyVtGUjYTAZhWkL6RhZx_adSEv/view?usp=sharing) extracted from Mask-RCNN;
- Download testing object tracks of [target videos](https://drive.google.com/file/d/1jP_kYzZbWuSHXox21o8doe-qqAERQWKP/view?usp=sharing) and [reference videos](https://drive.google.com/file/d/1kRh5t9k0U6v_2yZEMWvtPdPPpF6hp4H7/view?usp=sharing) extracted from Mask-RCNN.

### Run experiments on physical property learning
Train mass predictor
```
sh scripts/train_mass_property_prp.sh
```
Train charge predictor
```
sh scripts/train_charge_property_prp.sh
```
Evaluate on validation set of  mass predictor
```
sh scripts/test_mass_property.sh
```
Evaluate on validation set of charge predictor
```
sh scripts/test_charge_property.sh
```

### Run experiments on dynamic prediction
process output of physical property learner for dynamic predictor
```
sh scripts/post_processing_prp.sh
```
Train dynamic predictor
```
sh scripts/train_dynamic_predictor.sh
```
Test on dynamic predictor
```
sh scripts/test_dynamic_predictor.sh
```

### Acknowledgement
Much code is borrowed from Thomas Kipf's [NRI repo](https://github.com/ethanfetaya/NRI.git).
