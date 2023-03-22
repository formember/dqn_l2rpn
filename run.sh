n_actors=4
# Launch the Learner:

python learner_main.py --actor_update_period 300 \
  --taskstr gym,MountainCarContinuous-v0 --model_str 2048,2048,2048\
  --num_episodes 100000 --quantize_communication 32 --quantize 8\
  --replay_table_max_times_sampled 16 --batch_size 256 \
  --n_actors 4 --logpath logs/logfiles/ \
  --weight_compress 0 > logs/logs_stdout/learner_out

master_pid=$!
# Launch the Parameter Server:
python parameter_broadcaster.py --actor_update_period 300\
  --taskstr gym,MountainCarContinuous-v0 --model_str 2048,2048,2048\
  --num_episodes 100000 --quantize_communication 32 \
  --quantize 8 --replay_table_max_times_sampled 16 \
  --batch_size 256 --n_actors 4 --logpath logs/logfiles/ \
  --weight_compress 0 > logs/logs_stdout/broadcaster_out

# Launch the actors:
for i in `seq 1 $n_actors`; do
    python actor_main.py --actor_update_period 300 \
     --taskstr gym,MountainCarContinuous-v0  --model_str 2048,2048,2048 \
     --num_episodes 100000  --n_actors 4  --quantize_communication 32 \
     --quantize 8 --replay_table_max_times_sampled 16 --batch_size 256 --actor_id $i \
     --logpath logs/logfiles/ --weight_compress 0  > logs/logs_stdout/actorid=${i}_out &
    echo $!
done

# Wait
wait master_pid