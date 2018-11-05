# Deep Embedding Clustering (DEC)

Keras implementation for ICML-2016 paper:

* Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. ICML 2016.

## Usage
1. Install [Keras>=2.0.9](https://github.com/fchollet/keras), scikit-learn  
```
pip install keras scikit-learn   
```
2. Clone the code to local.   
```
git clone https://github.com/XifengGuo/DEC-keras.git DEC
cd DEC
```
3. Prepare datasets.    

Download **STL**:
```
cd data/stl
bash get_data.sh
cd ../..
```
**MNIST** and **Fashion-MNIST (FMNIST)** can be downloaded automatically when you run the code.

**Reuters** and **USPS**: If you cannot find these datasets yourself, you can download them from:   
https://pan.baidu.com/s/1hsMQ8Tm (password: `4ss4`) for **Reuters**, and  
https://pan.baidu.com/s/1skRg9Dr (password: `sc58`) for **USPS**


4. Run experiment on MNIST.   
`python DEC.py --dataset mnist`   
or (if there's pretrained autoencoder weights)  
The DEC model will be saved to "results/DEC_model_final.h5".

5. Other usages.   

Use `python DEC.py -h` for help.

## Results

```
python run_exp.py
```
Table 1. Mean performance over 10 trials. See [results.csv](./results/exp1/results.csv) for detailed results for each trial.  

   |        |     |kmeans|AE+kmeans|  DEC  |  paper    
   :--------|:---:|:----:|:-------:|:-----:|----:
   |mnist   | acc | 53   | 88      | 91    | 84 
   |        | nmi | 50   | 81      | 87    | --
   |fmnist  | acc | 47   | 61      | 62    | --
   |        | nmi | 51   | 64      | 65    | --
   |usps    | acc | 67   | 71      | 76    | --
   |        | nmi | 63   | 68      | 79    | --
   |stl     | acc | 70   | 79      | 86    | --
   |        | nmi | 71   | 72      | 82    | --
   |reuters | acc | 52   | 76      | 78    | 72
   |        | nmi | 31   | 52      | 57    | --

COIL20 (Reduced Size: 28x28) -- NMI: 0.8006; ACC: 0.7927

Iter 0: acc = 0.69097, nmi = 0.77883, ari = 0.61052  ; loss= 0
Iter 140: acc = 0.68819, nmi = 0.77565, ari = 0.61013  ; loss= 0.34916
Iter 280: acc = 0.69583, nmi = 0.78196, ari = 0.61927  ; loss= 0.24986
Iter 420: acc = 0.69306, nmi = 0.78575, ari = 0.61665  ; loss= 0.46839
Iter 560: acc = 0.69861, nmi = 0.79484, ari = 0.62703  ; loss= 0.08938
Iter 700: acc = 0.69792, nmi = 0.79352, ari = 0.62584  ; loss= 0.2613
Iter 840: acc = 0.69583, nmi = 0.79320, ari = 0.62423  ; loss= 0.31967
Iter 980: acc = 0.69653, nmi = 0.79431, ari = 0.62364  ; loss= 0.17717
Iter 1120: acc = 0.69722, nmi = 0.79510, ari = 0.62548  ; loss= 0.28154
Iter 1260: acc = 0.69931, nmi = 0.79606, ari = 0.62633  ; loss= 0.19733
Iter 1400: acc = 0.69861, nmi = 0.79689, ari = 0.62773  ; loss= 0.17857
Iter 1540: acc = 0.69792, nmi = 0.79628, ari = 0.62728  ; loss= 0.20577
Iter 1680: acc = 0.69861, nmi = 0.79759, ari = 0.62791  ; loss= 0.31679
Iter 1820: acc = 0.69931, nmi = 0.79771, ari = 0.62806  ; loss= 0.11897
Iter 1960: acc = 0.69931, nmi = 0.79780, ari = 0.62754  ; loss= 0.18506
Iter 2100: acc = 0.70000, nmi = 0.79857, ari = 0.62798  ; loss= 0.19794
Iter 2240: acc = 0.70000, nmi = 0.79760, ari = 0.62838  ; loss= 0.19827
Iter 2380: acc = 0.70000, nmi = 0.79804, ari = 0.62900  ; loss= 0.2144
Iter 2520: acc = 0.70069, nmi = 0.79928, ari = 0.62944  ; loss= 0.1796
Iter 2660: acc = 0.70208, nmi = 0.80000, ari = 0.63092  ; loss= 0.17944
Iter 2800: acc = 0.70139, nmi = 0.79835, ari = 0.62969  ; loss= 0.17534
Iter 2940: acc = 0.70278, nmi = 0.80021, ari = 0.63232  ; loss= 0.18781
Iter 3080: acc = 0.70208, nmi = 0.80029, ari = 0.63175  ; loss= 0.17931
Iter 3220: acc = 0.70278, nmi = 0.80060, ari = 0.63210  ; loss= 0.15894
Iter 3360: acc = 0.70278, nmi = 0.80060, ari = 0.63210  ; loss= 0.16314

## Autoencoder model

![](autoencoders.png)

## Other implementations

Original code (Caffe): https://github.com/piiswrong/dec   
MXNet implementation: https://github.com/dmlc/mxnet/blob/master/example/dec/dec.py   
Keras implementation without pretraining code: https://github.com/fferroni/DEC-Keras
