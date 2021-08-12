#!/bin/bash
CODEDIR=$1
RESULTSDIR=$2
SEED=$3

cd $CODEDIR

export PYTHONUSERBASE=$SCRATCHDIR
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.8/site-packages:$PYTHONPATH
pip install nvidia-ml-py3==7.352.0
pip install bayesian-optimization==1.1.0
pip install scipy==1.4.1
pip install pandas==1.0.5
pip install matplotlib==3.1.3
pip install torchcontrib==0.0.2
pip install wfdb==3.3.0


python team_code.py $RESULTSDIR $SEED
python test_run_code.py $RESULTSDIR $SEED
python run_evaluation.py $RESULTSDIR $SEED