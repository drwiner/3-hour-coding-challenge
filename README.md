# 3 Hour Python Coding Challenge 
A test for sample work from David R. Winer (drwiner131 at gmail)

This work was done in less than 3 hours as part of a coding test. 

Assignment: Write a general purpose decision tree classifier from scratch without using any machine learning libraries, ChatGPT, or code from others. Use the attached toy data set file to try out your classifier to predict the name of an animal based on its number of legs and color.

## Setup:
#### Using python 3.9.6
```
pip install -r requirements.txt
```

#### Set python path to be root folder
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

## Inference:
```
python run_inference.py --test_csv resources/animals.csv --model_dir temp_dir --out_dir output_dir 
```

```commandline 
python run_inference.py -h
```


#### Run inference with evaluation if target column is available
```
python run_inference.py --input_csv resources/test.csv --model_dir temp_dir --do_eval
```

# Test:
```commandline
pytest tests/test_run_training.py::test_main
```

TODO: testing for inference and "end to end".
