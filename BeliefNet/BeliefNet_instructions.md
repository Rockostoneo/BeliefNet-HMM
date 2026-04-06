libraries

1. edit the config json file. 
2. Train the model, you can override config setting with click commands `--name value`
```
python training.py config_example.json --mode train
```

3. To plot training and eval loss trajectory
```
python training.py config_example.json --mode visual
```

5. To extract the learned parameters (transition, emission, initial state) run
```
python training.py config_example.json --mode get_matricies 
```

6. To evalute a trained model on a new dataset, edit the evaluate.py file settings and run
```
python evaluate.py
```