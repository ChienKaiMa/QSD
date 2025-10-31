# https://unix.stackexchange.com/a/671401/552882
max_jobs=20 # set queue size
for ((q = 6; q < 16; q++)); do
    for ((n = 3; n < 20; n++)); do
        for ((seed = 0; seed < 5; seed++)); do
            num_jobs=$(pgrep -c -P$$)
            if [[ $num_jobs -ge $max_jobs ]]; then
                wait -n $(pgrep -P$$) # Wait until a any subprocess terminates
            else
                python3 solvers.py -n $n -q $q -s $seed &
            fi
        done
    done
done
