## Quick Start
This project is built upon the veRL(https://github.com/verl-project/verl) framework, an open-source toolkit for reinforcement learning.
```python
conda create -n verl python=3.12
cd verl
pip install -r requirements.txt
pip show torch  # Should be version 2.6 
pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
# Test import
python -c "import verl; print('VERL installed successfully')"
```

## Train
```python
cd verl
bash run_qwen3-8b.sh
```
