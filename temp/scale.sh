touch scale.log
for ((q = 2 ; q < 11 ; q++)); do
    for ((seed = 0 ; seed < 10 ; seed++)); do
        python3 build_circuits.py -n 3 -q $q -s $seed &>> scale.log
    done
done
