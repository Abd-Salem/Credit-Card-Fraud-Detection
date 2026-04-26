#!/bin/bash
set -e   # stop on first failure

# -s:   sampling technique[none, rus, enn, smote, smoteenn, smotetomek]            default=smote
# -alg: model algorithm[lr, rf, nn, nn_fl, vc1, vc2, vc3]                          default=lr
# -m:   mode[train, eval, full]                                                    default=full
# -p:   show plot[true, false]                                                     default=false

python main.py -s rus         -alg lr   -m eval   -p false
python main.py -s enn         -alg lr   -m eval   -p false
python main.py -s smote       -alg lr   -m eval   -p false
python main.py -s smoteenn    -alg lr   -m eval   -p false
python main.py -s smotetomek  -alg lr   -m eval   -p false

python main.py -s rus         -alg rf   -m eval   -p false
python main.py -s enn         -alg rf   -m eval   -p false
python main.py -s smote       -alg rf   -m eval   -p false
python main.py -s smoteenn    -alg rf   -m eval   -p false
python main.py -s smotetomek  -alg rf   -m eval   -p false

python main.py -s rus         -alg nn   -m eval   -p false
python main.py -s enn         -alg nn   -m eval   -p false
python main.py -s smote       -alg nn   -m eval   -p false
python main.py -s smoteenn    -alg nn   -m eval   -p false
python main.py -s smotetomek  -alg nn   -m eval   -p false

python main.py -s rus         -alg knn   -m eval    -p false
python main.py -s enn         -alg knn   -m eval    -p false
python main.py -s smote       -alg knn   -m eval    -p false
python main.py -s smoteenn    -alg knn   -m eval    -p false
python main.py -s smotetomek  -alg knn   -m eval    -p false

python main.py -s rus         -alg vc1   -m eval   -p false
python main.py -s enn         -alg vc1   -m eval   -p false
python main.py -s smote       -alg vc1   -m eval   -p false
python main.py -s smoteenn    -alg vc1   -m eval   -p false
python main.py -s smotetomek  -alg vc1   -m eval   -p false

python main.py -s rus         -alg vc2   -m eval   -p false
python main.py -s enn         -alg vc2   -m eval   -p false
python main.py -s smote       -alg vc2   -m eval   -p false
python main.py -s smoteenn    -alg vc2   -m eval   -p false
python main.py -s smotetomek  -alg vc2   -m eval   -p false

python main.py -s rus         -alg vc3   -m eval   -p false
python main.py -s enn         -alg vc3   -m eval   -p false
python main.py -s smote       -alg vc3   -m eval   -p false
python main.py -s smoteenn    -alg vc3   -m eval   -p false
python main.py -s smotetomek  -alg vc3   -m eval   -p false

python main.py -s rus         -alg nn_fl   -m eval    -p false
python main.py -s enn         -alg nn_fl   -m eval    -p false
python main.py -s smote       -alg nn_fl   -m eval    -p false
python main.py -s smoteenn    -alg nn_fl   -m eval    -p false
python main.py -s smotetomek  -alg nn_fl   -m eval    -p false