python -m cProfile -s cumtime -o profile_output.prof experiments.py
# python -c "import pstats; p = pstats.Stats('profile_output.prof'); p.sort_stats('cumulative').print_stats()" > profile_readable.txt
python -c "import pstats; p = pstats.Stats('profile_output.prof'); p.sort_stats('tottime').print_stats()" > profile_readable.txt
python -c "import pstats; p = pstats.Stats('profile_output.prof'); p.sort_stats('time').print_stats()" > profile_readable.txt
