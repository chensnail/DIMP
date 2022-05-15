
nohup python -u execute.py --dataset cora --nb_epochs 2000 --patience 20 --numb_start 4 --numb_end 5 --chu_start 1 --chu_end 2 --chu_strip 0.5 --h1 -0.1 --h2 -0.1 --k_numb1 0 --k_numb2 1 >yyy.log 2>&1 &
nohup python -u execute.py --dataset pubmed --nb_epochs 2000 --patience 20 --numb_start 4 --numb_end 5 --chu_start 0.9 --chu_end 1 --chu_strip 0.1 --h1 -0.1 --h2 -0.1 --k_numb1 0 --k_numb2 1 >yyy.log 2>&1 &
nohup python -u execute.py --dataset citeseer --nb_epochs 20 --patience 20 --numb_start 4 --numb_end 5 --chu_start 3 --chu_end 3.1 --chu_strip 0.1 --h1 -0.5 --h2 -0.2 --k_numb1 1 --k_numb2 0 >yyy.log 2>&1 &

nohup python -u execute.py --dataset ms_academic_phy --nb_epochs 200 --patience 25 --numb_start 4 --numb_end 5 --chu_start 1 --chu_end 2 --chu_strip 0.5 --h1 -0.1 --h2 -0.1 --k_numb1 1 --k_numb2 0 >yyy.log 2>&1 &
nohup python -u execute.py --dataset ms_academic_cs --nb_epochs 300 --patience 25 --numb_start 4 --numb_end 5 --chu_start 2.6 --chu_end 2.8 --chu_strip 0.2 --h1 -0.1 --h2 -0.1 --k_numb1 1 --k_numb2 0 >yyy.log 2>&1 &

nohup python -u execute.py --dataset amazon_electronics_photo --nb_epochs 200 --patience 20 --numb_start 4 --numb_end 5 --chu_start 3.5 --chu_end 4 --chu_strip 0.5 --h1 -0.3 --h2 -0.3 --k_numb1 1 --k_numb2 0 >yyy.log 2>&1 &
nohup python -u execute.py --dataset amazon_electronics_computers --nb_epochs 300 --patience 25 --numb_start 4 --numb_end 5 --chu_start 1.6 --chu_end 1.7 --chu_strip 0.1 --h1 -0.2 --h2 -0.2 --k_numb1 0 --k_numb2 1 >yyy.log 2>&1 &

