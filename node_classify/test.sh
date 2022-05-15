#!/usr/bin/env bash
while  read line
  do
      #echo $line
      for((numb=4;numb<=5;numb=numb+1))
        do
          for((chu=1;chu<=4;chu=chu+0.2))
            do
              for((hh=0.1;hh<=0.5;hh=hh+0.1))
                do
                  echo >
                  for((ep=1;ep<=20;ep=ep+1))
                    do
                      nohup python -u execute.py --dataset $line --nb_epochs 2000 --patience 20 --numb $numb --chu $chu --h1 -$hh --h2 -$hh --k_numb1 0 --k_numb2 1 >yyy.log 2>&1 & > yyy.log 2>&1 &
                    done
                done
            done
        done
  done < data_name.txt
