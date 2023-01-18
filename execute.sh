python3 trainer.py --epoch 50 --da --arch efficientnetb4 --batch 16 --opt Adam --clr triangular2 --clr triangular2 --step 4

python3 trainer.py --epoch 50 --da --arch frozenefficientnetb4 --batch 16 --opt Adam --clr triangular --dropout 0.1 --clr triangular2 --step 4