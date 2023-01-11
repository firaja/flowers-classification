#python3 trainer.py --epoch 6 --da --mp --arch vgg16 --batch 16 --step 2 --opt SGD --clr triangular
#python3 trainer.py --epoch 6 --mp --arch vgg16 --batch 16 --step 2 --opt SGD --clr triangular
#python3 trainer.py --epoch 6 --da --mp --arch vgg16 --batch 16 --step 2 --opt Adam --clr triangular 
#python3 trainer.py --epoch 6 --mp --arch vgg16 --batch 16 --step 2 --opt Adam --clr triangular
#python3 trainer.py --epoch 6 --da --mp --arch vgg16 --batch 16 --step 4 --opt SGD --clr triangular
#python3 trainer.py --epoch 6 --mp --arch vgg16 --batch 16 --step 4 --opt SGD --clr triangular
#python3 trainer.py --epoch 6 --da --mp --arch vgg16 --batch 16 --step 4 --opt Adam --clr triangular 
#python3 trainer.py --epoch 6 --mp --arch vgg16 --batch 16 --step 4 --opt Adam --clr triangular

#python3 trainer.py --epoch 6 --mp --arch resnet18 --batch 16 --step 4 --opt SGD --clr triangular
#python3 trainer.py --epoch 6 --da --mp --arch resnet18 --batch 16 --step 4 --opt SGD --clr triangular
#python3 trainer.py --epoch 6 --mp --arch resnet18 --batch 16 --step 4 --opt Adam --clr triangular 
#python3 trainer.py --epoch 6 --da --mp --arch resnet18 --batch 16 --step 4 --opt Adam --clr triangular
#python3 trainer.py --epoch 6 --mp --arch resnet18 --batch 16 --step 4 --opt SGD --clr triangular
#python3 trainer.py --epoch 6 --da --mp --arch resnet18 --batch 16 --step 4 --opt SGD --clr triangular
#python3 trainer.py --epoch 6 --mp --arch resnet18 --batch 16 --step 4 --opt Adam --clr triangular 
#python3 trainer.py --epoch 6 --da --mp --arch resnet18 --batch 16 --step 4 --opt Adam --clr triangular

python3 trainer.py --epoch 6 --da --arch inceptionv3 --batch 16 --opt Adam --clr triangular
python3 trainer.py --epoch 6 --da --arch inceptionv3 --batch 8 --opt Adam --clr triangular
python3 trainer.py --epoch 6 --da --arch inceptionv3 --batch 4 --opt Adam --clr triangular
python3 trainer.py --epoch 6 --da --arch inceptionv3 --batch 16 --opt SGD --clr triangular
python3 trainer.py --epoch 6 --da --arch inceptionv3 --batch 8 --opt SGD --clr triangular
python3 trainer.py --epoch 6 --da --arch inceptionv3 --batch 4 --opt SGD --clr triangular