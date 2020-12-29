# GIRTH MCMC
Item Response Theory using Markov Chain Monte Carlo

**Dependencies**

We recommend using [Anaconda](https://www.anaconda.com/products/individual). Individual
packages can be installed through pip otherwise.

* Python >= 3.7.6
* Numpy
* Scipy
* PyMC3

# Supports
**Unidimensional**
* Rasch Model 
* 1PL Model
* 2PL Model
* 3PL Model
* Graded Response Model

# Usage
Subject to change but for now:
```python
discrimination = 0.89 * np.sqrt(-2 * np.log(np.random.rand(10)))
difficulty = np.random.randn(10)
theta = np.random.randn(100)

syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination, 
                                            theta)

girth_model = GirthMCMC(model='2PL', 
                        options={'n_processors': 4})
results = girth_model(syn_data)
print(results)
```

for the graded response model, pass in the number of categories
```python
n_categories = 3

difficulty = np.random.randn(10, n_categories-1)
difficulty = np.sort(difficulty, 1)        
discrimination = 0.96 * np.sqrt(-2 * np.log(np.random.rand(10)))
theta = np.random.randn(150)

syn_data = create_synthetic_irt_polytomous(difficulty, discrimination, 
                                            theta, model='grm')

girth_model = GirthMCMC(model='GRM', model_args=(n_categories,),
                        options={'n_processors': 4})
results = girth_model(syn_data)
print(results)
```
## Other Estimation Packages
If you are looking for Marginal Maximum Likelihood estimation routines, 
check out [GIRTH](https://eribean.github.io/girth/).
