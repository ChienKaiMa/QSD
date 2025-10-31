# script
# Try different initial states
# May have strange issues
# https://unix.stackexchange.com/a/671401/552882
max_jobs=20 # set queue size
for ((q = 2; q < 6; q++)); do
    for ((n = 3; n < 4; n++)); do
        for ((seed = 0; seed < 5; seed++)); do
            num_jobs=$(pgrep -c -P$$)
            if [[ $num_jobs -ge $max_jobs ]]; then
                wait -n $(pgrep -P$$) # Wait until a any subprocess terminates
            else
                python -m cProfile -s cumtime -o "profile_output_q${q}_n${n}_s${seed}.prof" flow/experiments.py -n $n -q $q -s $seed &> "20250223_q${q}_n${n}_s${seed}.log" &
            fi
        done
    done
done
