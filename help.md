usage: main.py recognition [-h] [-w WORK_DIR] [-c CONFIG] [--phase PHASE]
                           [--save_result SAVE_RESULT]
                           [--start_epoch START_EPOCH] [--num_epoch NUM_EPOCH]
                           [--use_gpu USE_GPU] [--device DEVICE [DEVICE ...]]
                           [--log_interval LOG_INTERVAL]
                           [--save_interval SAVE_INTERVAL]
                           [--eval_interval EVAL_INTERVAL]
                           [--save_log SAVE_LOG] [--print_log PRINT_LOG]
                           [--pavi_log PAVI_LOG] [--feeder FEEDER]
                           [--num_worker NUM_WORKER]
                           [--train_feeder_args TRAIN_FEEDER_ARGS]
                           [--test_feeder_args TEST_FEEDER_ARGS]
                           [--batch_size BATCH_SIZE]
                           [--test_batch_size TEST_BATCH_SIZE] [--debug]
                           [--model MODEL] [--model_args MODEL_ARGS]
                           [--weights WEIGHTS]
                           [--ignore_weights IGNORE_WEIGHTS [IGNORE_WEIGHTS ...]]
                           [--show_topk SHOW_TOPK [SHOW_TOPK ...]]
                           [--base_lr BASE_LR] [--step STEP [STEP ...]]
                           [--optimizer OPTIMIZER] [--nesterov NESTEROV]
                           [--weight_decay WEIGHT_DECAY]

optional arguments:
  -h, --help            show this help message and exit
  -w WORK_DIR, --work_dir WORK_DIR
                        the work folder for storing results
  -c CONFIG, --config CONFIG
                        path to the configuration file
  --phase PHASE         must be train or test
  --save_result SAVE_RESULT
                        if ture, the output of the model will be stored
  --start_epoch START_EPOCH
                        start training from which epoch
  --num_epoch NUM_EPOCH
                        stop training in which epoch
  --use_gpu USE_GPU     use GPUs or not
  --device DEVICE [DEVICE ...]
                        the indexes of GPUs for training or testing
  --log_interval LOG_INTERVAL
                        the interval for printing messages (#iteration)
  --save_interval SAVE_INTERVAL
                        the interval for storing models (#iteration)
  --eval_interval EVAL_INTERVAL
                        the interval for evaluating models (#iteration)
  --save_log SAVE_LOG   save logging or not
  --print_log PRINT_LOG
                        print logging or not
  --pavi_log PAVI_LOG   logging on pavi or not
  --feeder FEEDER       data loader will be used
  --num_worker NUM_WORKER
                        the number of worker per gpu for data loader
  --train_feeder_args TRAIN_FEEDER_ARGS
                        the arguments of data loader for training
  --test_feeder_args TEST_FEEDER_ARGS
                        the arguments of data loader for test
  --batch_size BATCH_SIZE
                        training batch size
  --test_batch_size TEST_BATCH_SIZE
                        test batch size
  --debug               less data, faster loading
  --model MODEL         the model will be used
  --model_args MODEL_ARGS
                        the arguments of model
  --weights WEIGHTS     the weights for network initialization
  --ignore_weights IGNORE_WEIGHTS [IGNORE_WEIGHTS ...]
                        the name of weights which will be ignored in the
                        initialization
  --show_topk SHOW_TOPK [SHOW_TOPK ...]
                        which Top K accuracy will be shown
  --base_lr BASE_LR     initial learning rate
  --step STEP [STEP ...]
                        the epoch where optimizer reduce the learning rate
  --optimizer OPTIMIZER
                        type of optimizer
  --nesterov NESTEROV   use nesterov or not
  --weight_decay WEIGHT_DECAY
                        weight decay for optimizer
