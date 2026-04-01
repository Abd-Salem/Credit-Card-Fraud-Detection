#!/bin/bash
set -e   # stop on first failure

# -s:   sampling technique[none, rus, enn, smote, smoteenn, smotetomek]            default=smote
# -alg: model algorithm[lr, rf, nn, nn_fl]                                         default=lr
# -m:   mode[train, eval, full]                                                    default=full
# -p:   show plot[true, false]                                                     default=false

python main.py -s rus         -alg lr
python main.py -s enn         -alg lr
python main.py -s smote       -alg lr
python main.py -s smoteenn    -alg lr
python main.py -s smotetomek  -alg lr

python main.py -s rus         -alg rf
python main.py -s enn         -alg rf
python main.py -s smote       -alg rf
python main.py -s smoteenn    -alg rf
python main.py -s smotetomek  -alg rf

python main.py -s rus         -alg nn
python main.py -s enn         -alg nn
python main.py -s smote       -alg nn
python main.py -s smoteenn    -alg nn
python main.py -s smotetomek  -alg nn

python main.py -s rus        -alg nn_fl
python main.py -s enn        -alg nn_fl
python main.py -s smote      -alg nn_fl
python main.py -s smoteenn   -alg nn_fl
python main.py -s smotetomek -alg nn_fl

python main.py -s rus        -alg knn
python main.py -s enn        -alg knn
python main.py -s smote      -alg knn
python main.py -s smoteenn   -alg knn
python main.py -s smotetomek -alg knn

python main.py -s rus        -alg vc
python main.py -s enn        -alg vc
python main.py -s smote      -alg vc
python main.py -s smoteenn   -alg vc
python main.py -s smotetomek -alg vc