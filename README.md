# Comprehensible Score based model 

This repository explores the new class of generative models named score based models. The key idea is to model the gradient of the log probability density function, a quantity often known as the **score function**. This class of models is inspired by considerations from thermodynamics, but also bears strong resemblance to _denoising score matching_ , _Langevin dynamics_ and _autoregressive decoding_. 

You will find a introductory pdf about score based model in : 'score-based.pdf'

### Implementation

I also provide a very simple implementation of score based model with the most common score matching techniques. The source code of the implementation is located in 'src/' file. 

Please make sure that you have python 3+ install in your machine, you will also have to install dependencies which are:
- torch
- numpy 
- matplotlib
- sklearn

To run the model and plot your results, use the following commands :

```bash
cd src/
python model.py
```

Results will be saved in the 'img/' file.
