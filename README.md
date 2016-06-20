# pacmanCTF

## Training

### Offensive Q Agent
for example, Red team as offender, Blue team as defender, train for 5 times and play for 2 times, the weights will store in offensive.train

`python capture.py -r superTeam --keys0 --keys1 -x 5 -n 7`

### Deffensive Q Agent

## Testing
If need to use pretrain weights, need to uncomment some lines in `__init__`, otherwise, best weight is use

`python capture.py -r superTeam`
