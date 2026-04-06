activate venv

To train a model, adjust the config file in the `\config` folder to desired settings, then
```
python train.py config/train_m32.py
```

To just get final validation/test loss over all data (not just a single batch). Edit the two lines in `validation.py`, then 
```
python validation.py
```

