# Learning-to-Detect-Malicious-Clients-for-Robust-FL

Here, we present part of the experimental codes in the paper [Learning to Detect Malicious Clients for Robust Federated Learning](https://arxiv.org/abs/2002.00211). The codes presented here correspond to the experiments on the MNIST dataset.


## Prerequisites
Our experiments were run on AWS EC2 GPU instances (type p2.xlarge). We use Python 3.6 and the required packages are listed as follows.

* PyTorch         1.4.0
* torchvision     0.5.0
* scikit-learn    0.21.3 
* skimage         0.15.0        
* numpy           1.17.2
* scipy           1.3.1
* matplotlib      3.1.1
* pandas          0.25.1
* hdmedians       0.13

Please also download the detection model of the MNIST dataset from the anonymous link:
https://drive.google.com/file/d/1rPiOPGvFO_CwaoboBIf38n-9ztchCzWD/view?usp=sharing

We also provide a ./requirements.txt file for your reference.


## How to reproduce results:
We provide command for each experimental case. All intermediate results are printed and redirected to a file.

These commands shall be run in ./src/ directory.

===With Vanilla FedAvg Method===

Case (No attack):
nohup python -u main_regression.py > nohup_mnist_no_attack_fedavg.log 2>&1 &

Case (Sign-flipping attack, attacker ratio 0.3):
nohup python -u main_regression.py  --attack_ratio 0.3 --attack_mode="sign" > nohup_mnist_signflipping0.3_fedavg.log 2>&1 &

Case (Sign-flipping attack, attacker ratio 0.5):
nohup python -u main_regression.py  --attack_ratio 0.5 --attack_mode="sign" > nohup_mnist_signflipping0.3_fedavg.log 2>&1 &

Case (Additive attack, attacker ratio 0.3):
nohup python -u main_regression.py  --attack_ratio 0.3 --attack_mode="noise" > nohup_mnist_noise0.3_fedavg.log 2>&1 &

Case (Additive attack, attacker ratio 0.5):
nohup python -u main_regression.py  --attack_ratio 0.5 --attack_mode="noise" > nohup_mnist_noise0.5_fedavg.log 2>&1 &

Case (Targeted attack):
nohup python -u main_regression_poison.py > nohup_mnist_targeted_fedavg.log 2>&1 &

===With GeoMed Method===

Case (No attack)
nohup python -u main_regression.py  --aggregation="GeoMed" > nohup_mnist_no_attack_GeoMed.log 2>&1 &

Case (Sign-flipping attack, attacker ratio 0.3):
nohup python -u main_regression.py  --attack_ratio 0.3 --attack_mode="sign" --aggregation="GeoMed" > nohup_mnist_sign0.3_GeoMed.log 2>&1 &

Case (Sign-flipping attack, attacker ratio 0.5)
nohup python -u main_regression.py  --attack_ratio 0.5 --attack_mode="sign" --aggregation="GeoMed" > nohup_mnist_sign0.5_GeoMed.log 2>&1 &

Case (Additive noise attack, attacker ratio 0.3)
nohup python -u main_regression.py  --attack_ratio 0.3 --attack_mode="noise" --aggregation="GeoMed" > nohup_mnist_noise0.3_GeoMed.log 2>&1 &

Case (Additive noise attack, attacker ratio 0.5)
nohup python -u main_regression.py  --attack_ratio 0.5 --attack_mode="noise" --aggregation="GeoMed" > nohup_mnist_noise0.5_GeoMed.log 2>&1 &

Case (Targeted attack)
nohup python -u main_regression_poison.py --aggregation="GeoMed" > nohup_mnist_targeted_GeoMed.log 2>&1 &

===With Krum Method===

Case (No attack)
nohup python -u main_regression.py  --aggregation="Krum" > nohup_mnist_no_attack_Krum.log 2>&1 &

Case (Sign-flipping attack, attacker ratio 0.3)
nohup python -u main_regression.py  --attack_ratio 0.3 --attack_mode="sign" --aggregation="Krum" > nohup_mnist_sign0.3_Krum.log 2>&1 &

Case (Sign-flipping attack, attacker ratio 0.5)
nohup python -u main_regression.py  --attack_ratio 0.5 --attack_mode="sign" --aggregation="Krum" > nohup_mnist_sign0.5_Krum.log 2>&1 &

Case (Additive noise attack, attacker ratio 0.3)
nohup python -u main_regression.py  --attack_ratio 0.3 --attack_mode="noise" --aggregation="Krum" > nohup_mnist_noise0.3_Krum.log 2>&1 &

Case (Additive noise attack, attacker ratio 0.5)
nohup python -u main_regression.py  --attack_ratio 0.5 --attack_mode="noise" --aggregation="Krum" > nohup_mnist_noise0.5_Krum.log 2>&1 &

Case (Targeted attack)
nohup python -u main_regression_poison.py --aggregation="Krum" > nohup_mnist_targeted_Krum.log 2>&1 &

===With Our Method===

Case (No attack)
nohup python -u main_regression.py  --aggregation="atten" --vae_model="mnist_vae_16100.pt" > nohup_mnist_no_attack_ours.log 2>&1 &

Case (Sign-flipping attack, attacker ratio 0.3)
nohup python -u main_regression.py  --attack_ratio 0.3 --attack_mode="sign" --aggregation="atten" --vae_model="mnist_vae_16100.pt" > nohup_mnist_sign0.3_ours.log 2>&1 &

Case (Sign-flipping attack, attacker ratio 0.5)
nohup python -u main_regression.py  --attack_ratio 0.5 --attack_mode="sign" --aggregation="atten" --vae_model="mnist_vae_16100.pt" > nohup_mnist_sign0.5_ours.log 2>&1 &

Case (Additive noise attack, attacker ratio 0.3)
nohup python -u main_regression.py  --attack_ratio 0.3 --attack_mode="noise" --aggregation="atten" --vae_model="mnist_vae_16100.pt" > nohup_mnist_noise0.3_ours.log 2>&1 &

Case (Additive noise attack, attacker ratio 0.5)
nohup python -u main_regression.py  --attack_ratio 0.5 --attack_mode="noise" --aggregation="atten" --vae_model="mnist_vae_16100.pt" > nohup_mnist_noise0.5_ours.log 2>&1 &

Case (Targeted attack)
nohup python -u main_regression_poison.py --aggregation="atten" --vae_model="mnist_vae_16100.pt" > nohup_mnist_targeted_ours.log 2>&1 &



## Comments of each file in ./codes directory:

Codes at current version are not organized very well. Therefore, we provide additional explanation of each file for your reference.

* ./src/aggregations.py: implement different defense methods, such as GeoMed, Krum and ours.
* ./src/attacks.py: implement untargarted attacks, including sign-flipping and additive noise
* ./src/main_regression.py: this script is for LR model on MNIST dataset. With different options, we obtain experimental results of LR model with untargeted attack.
* ./src/main_regression_poison.py: this script is for LR model on MNIST dataset. With different options, we obtain experimental results of LR model with targeted attack.

* ./src/util/sampling.py: implement sampling functions on MNIST dataset to generate non-iid data.
* ./src/models/Fed.py: implement vanilla FedAvg algorithm here.
* ./src/models/Nets.py: implement our logistic regression model and neural models here.
* ./src/models/Update.py: implement functions by which clients update their models here.
* ./src/models/test.py: implement functions by which we obtain model performance on testing set.

* ./src/models/mnist_vae_16100.pt: trained detection model for MNIST experiment. (should be downloaded)
