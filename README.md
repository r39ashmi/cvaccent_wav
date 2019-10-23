# cvaccent_wav
This repository contains data split, evaluation code along with trained models for five configurations of TFN:
(1) LearnFBANK (../models/learnFBANK.ckpt.tar)
(2) Fixed (../models/fixed.ckpt.tar)
(3) Learnall (../models/learnall.ckpt.tar)
(4) RandInt (../models/rand.ckpt.tar)
(5) LinearInt (../models/linear.ckpt.tar)

To evaluate follow the below given steps:

cd code

python3 evaluation.py <config-type> 
