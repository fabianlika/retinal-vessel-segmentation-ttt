@echo off
cd /d C:\Users\lenovo\source\repos\retinal-vessel-segmentation-ttt
set PY=C:\Users\lenovo\AppData\Local\Programs\Python\Python314\python.exe
set CKPT=checkpoints/best_model.pth
set STEPS=5
set LR=1e-6

echo ===== STARE ===== >> eval_results.log 2>&1
%PY% -u evaluation/evaluate.py --checkpoint %CKPT% --dataset stare --data_root data/STARE --img_size 512 --ttt_steps %STEPS% --ttt_lr %LR% >> eval_results.log 2>&1

echo ===== CHASE ===== >> eval_results.log 2>&1
%PY% -u evaluation/evaluate.py --checkpoint %CKPT% --dataset chase --data_root data/CHASE --img_size 512 --ttt_steps %STEPS% --ttt_lr %LR% >> eval_results.log 2>&1

echo ===== HRF ===== >> eval_results.log 2>&1
%PY% -u evaluation/evaluate.py --checkpoint %CKPT% --dataset hrf --data_root data/HRF --img_size 512 --ttt_steps %STEPS% --ttt_lr %LR% >> eval_results.log 2>&1

echo ===== ALL DONE ===== >> eval_results.log 2>&1
