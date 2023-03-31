# siamak-test

## Setup:
### Using python 3.9.6
```
pip install -r requirements.txt
```

### Consider setting python path to be root folder.


## Run:
#### Expected: run from root folder.

```
python run_training.py --input_csv resources/animals.csv --directory temp_dir --target_col Name
```

### Help:
```commandline 
python run_training.py -h
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
