# rm nohup.out
rm *.json
## -s choose from 'calls', 'cumtime', 'cumulative', 'filename', 'line', 'module',
## 'name', 'ncalls', 'nfl', 'pcalls', 'stdname', 'time', 'tottime'
python3 -m cProfile -o my_prof_file.out scripts/prof.py
# nohup python3 scripts/status.py &
nohup python -c "import pstats; p=pstats.Stats('my_prof_file.out'); p.sort_stats('time').print_stats()" &
sleep 3
mv nohup.out my_prof.out 
rm my_prof_file.out
