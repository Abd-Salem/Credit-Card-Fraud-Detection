#!/bin/bash
set -e   # stop on first failure

# -s:   sampling technique[none, rus, enn, smote, smoteenn, smotetomek]            default=smote
# -alg: model algorithm[lr, rf, nn, nn_fl]                                         default=lr
# -m:   mode[train, eval, full]                                                    default=full
# -p:   show plot[true, false]                                                     default=false

python main.py -s rus         -alg lr   -m eval   -p true
python main.py -s enn         -alg lr   -m eval   -p true
python main.py -s smote       -alg lr   -m eval   -p true
python main.py -s smoteenn    -alg lr   -m eval   -p true
python main.py -s smotetomek  -alg lr   -m eval   -p true

python main.py -s rus         -alg rf   -m eval   -p true
python main.py -s enn         -alg rf   -m eval   -p true
python main.py -s smote       -alg rf   -m eval   -p true
python main.py -s smoteenn    -alg rf   -m eval   -p true
python main.py -s smotetomek  -alg rf   -m eval   -p true

python main.py -s rus         -alg nn   -m eval   -p true
python main.py -s enn         -alg nn   -m eval   -p true
python main.py -s smote       -alg nn   -m eval   -p true
python main.py -s smoteenn    -alg nn   -m eval   -p true
python main.py -s smotetomek  -alg nn   -m eval   -p true

python main.py -s rus         -alg nn_fl   -m eval    -p true
python main.py -s enn         -alg nn_fl   -m eval    -p true
python main.py -s smote       -alg nn_fl   -m eval    -p true
python main.py -s smoteenn    -alg nn_fl   -m eval    -p true
python main.py -s smotetomek  -alg nn_fl   -m eval    -p true