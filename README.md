# siamak-test

## Setup:
### Using python 3.9.6
```
pip install -r requirements.txt
```

## Set python path to be root folder
```commandline
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Run:
#### Expected: run from root folder.

```
python run_training.py --input_csv resources/animals.csv --directory temp_dir --target_col Name
```

### Help:
```commandline 
python run_training.py -h
```

### Test:
```commandline
pytest tests/test_run_training.py::test_main
```

## Inference:
#### Expected: run from root folder.
```
python run_inference.py --input_csv resources/test.csv --model_dir temp_dir 
```

```commandline 
python run_inference.py -h
```


#### Run inference with evaluation if target column is available
```
python run_inference.py --input_csv resources/test.csv --model_dir temp_dir --target_col Name
```
