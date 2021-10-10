[![CodeFactor](https://www.codefactor.io/repository/github/eribean/girth_mcmc/badge)](https://www.codefactor.io/repository/github/eribean/girth_mcmc)
[![PyPI version](https://badge.fury.io/py/girth-mcmc.svg)](https://badge.fury.io/py/girth-mcmc)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

# GIRTH MCMC
Item Response Theory using Markov Chain Monte Carlo / Variational Inference

**Dependencies**

We recommend using [Anaconda](https://www.anaconda.com/products/individual). Individual
packages can be installed through pip otherwise.

* Python >= 3.7.6
* Numpy
* Scipy
* Girth
* PyMC3

## Installation

Via pip

```sh
pip install girth_mcmc --upgrade
```

From Source

```sh
pip install . -t $PYTHONPATH --upgrade
```

# Supports

**Unidimensional**
* Rasch Model 
* 1PL Model
* 2PL Model
* 3PL Model
* Graded Response Model

**Multi-dimensional**
* 2PL Model

# Usage

Subject to change but for now:

```python
import numpy as np
from girth import create_synthetic_irt_dichotomous
from girth_mcmc import GirthMCMC
                        
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
import numpy as np
from girth import create_synthetic_irt_polytomous
from girth_mcmc import GirthMCMC

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

Is some data missing? Tag it with a convenience function and run it like normal

```python
import numpy as np
from girth import create_synthetic_irt_dichotomous
from girth_mcmc import GirthMCMC
from girth_mcmc.utils import tag_missing_data_mcmc
                        
discrimination = 0.89 * np.sqrt(-2 * np.log(np.random.rand(10)))
difficulty = np.random.randn(10)
theta = np.random.randn(100)

syn_data = create_synthetic_irt_dichotomous(difficulty, discrimination, 
                                            theta)
mask = np.random.rand(*syn_data.shape) < .1
syn_data[mask] = -9999
syn_data_missing = tag_missing_data_mcmc(syn_data, [0, 1])

girth_model = GirthMCMC(model='2PL', 
                        options={'n_processors': 4})
results = girth_model(syn_data_missing)
print(results)
```

Don't like waiting? me either. Run Variational Inference for faster
but less accurate estimation.

```python
import numpy as np
from girth import create_synthetic_irt_polytomous
from girth_mcmc import GirthMCMC

n_categories = 3

difficulty = np.random.randn(10, n_categories-1)
difficulty = np.sort(difficulty, 1)        
discrimination = 1.76 * np.sqrt(-2 * np.log(np.random.rand(10)))
theta = np.random.randn(150)

syn_data = create_synthetic_irt_polytomous(difficulty, discrimination, 
                                            theta, model='grm')

girth_model = GirthMCMC(model='GRM', model_args=(n_categories,),
                        options={'variational_inference': True,
                                 'variational_samples': 10000,
                                 'n_samples': 10000})
results_variational = girth_model(syn_data, progressbar=False)
print(results_variational)
```

## Unittests

**pytest** with coverage.py module

```sh
pytest --cov=girth_mcmc --cov-report term
```

## Contact

Ryan Sanchez  
ryan.sanchez@gofactr.com

## Other Estimation Packages

If you are looking for Marginal Maximum Likelihood estimation routines,
check out [GIRTH](https://eribean.github.io/girth/), a graphical interface
is also at [GoFactr](https://gofactr.com)

## License

MIT License

Copyright (c) 2021 Ryan C. Sanchez

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
