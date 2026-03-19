# ERR
## 准备工作
```python
conda create -n verl python=3.12
cd verl
pip install -r requirements.txt
pip show torch  #应该要是2.6，不是的话下面命令需要更改
pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl
# 测试导入
python -c "import verl; print('VERL installed successfully')"
```

## 运行训练
```python
cd verl
bash run_qwen3-8b.sh
```
