fifo_replay_4 shows loss of plasticity

fifo_replay_5 shows loss of plasticity with replay of size 2000

fifo_replay_6 shows loss of plasticity with replay of size 20000

fifo_replay_7 shows loss of plasticity with replay of size 60000


fifo_replay_8 shows loss of plasticity with replay of size 2000 with permutation on original image for every task

fifo_replay_9 shows loss of plasticity with replay of size 2000 with permutation on original image for every task, we see that prequential accuracy basically is masking the loss of plasticity due to low accuracy in
initial batches, this does not mean loss of plasticity is absent just that we couldn't clearly see it.


fifo_replay_10 shows little loss of plasticity with replay of size 2000 with permutation on original image for every task
we use weight decay of 0.0005 and forward accuracy calculated on current test set


code_v11: 
1. Buffer size 300000
2. Permutation on original data
3. No weight decay



code_v12: 
1. Buffer size 2000
2. Permutation on original data
3. No weight decay


==> everything before code_v11 to be deleted after results for code_v11 and code_v12 are seen