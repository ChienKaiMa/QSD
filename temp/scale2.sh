touch scale.log
for ((q = 2 ; q < 4 ; q++)); do
    for ((n = 3 ; n < (2**q - 1) ; n++)); do
        for ((seed = 0 ; seed < 5 ; seed++)); do
            python3 build_circuits.py -n $n -q $q -s $seed &>> scale2.log
        done
    done
done
