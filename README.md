# YOLOX_pruning
YOLOX model prune


### Train origin yolox-l model
```
python3 tools/train.py -f ../sample/my_hand_voc_l.py -b 32 --fp16
```

### Train sparsity yolox-l model
add depend to **tools/train.py**
```
import sys
from pathlib import Path 
sys.path.append(str(Path(__file__).resolve().parents[2]/'prun_train'))
from my_train import MY_Trainer_Loose
```
and then change **trainer = Trainer(exp, args)** to **trainer = MY_Trainer_Loose(exp, args)**
and then train the model
```
python3 tools/train.py -f ../sample/my_hand_voc_l.py -b 32 --fp16
```

### pruning and retrain
```
python3 ../prun_train/prune.py -c ~/Downloads/latest_ckpt.pth -f ../sample/my_hand_voc_l.py -expn pruned_model
```