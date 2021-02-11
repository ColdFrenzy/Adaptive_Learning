# Adaptive_Learning

# Installation guide

The code contained in this directory is not fully compatible with the current ray release (1.1.0). In order to use the
code before the release of the next version,
a [ray build from source](https://docs.ray.io/en/master/development.html#building-ray-python-only) is required.

**Installation example:**

* Pip install the latest Ray wheels:
  
  On windows
  
  ``` pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp36-cp36m-win_amd64.whl```
  
  On Ubuntu 
  
  ```pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-2.0.0.dev0-cp37-cp37m-manylinux2014_x86_64.whl```
  
* Clone my ray branch in your local directory:

  ` git clone https://github.com/coldfrenzy/ray.git `

* If you already have a ray version installed, replace Python files in the installed package with your local editable
  copy using the following code:

```
    cd ray 
    python python/ray/setup-dev.py 
```





 
