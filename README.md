# dslr-py

## Install with pip

```sh
pip install --upgrade git+https://github.com/audriusrudalevicius/dslr-py
```


## Usage

```sh
dslr_py src/ out/

# One gpu lock with file partitioning
CUDA_VISIBLE_DEVICES="1" dslrs_py --device='/gpu:1' --partition='1/2' src/ out/
```
