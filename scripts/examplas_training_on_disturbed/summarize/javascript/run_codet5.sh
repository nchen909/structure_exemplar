WORKDIR="/root/autodl-tmp/HugCode"
HUGGINGFACE_LOCALS="/root/autodl-tmp/HugCode/data/huggingface_models/"
# ORIGIN_MODEL_DIR="/data/huggingface_models/"
ORIGIN_MODEL_DIR="/root/autodl-tmp/HugCode/save_models/disturb"
export PYTHONPATH=$WORKDIR

#3090()
CUDA=7
BATCH_SIZE=16
DEV_BATCH_SIZE=16
TEST_BATCH_SIZE=16
NUM_TRAIN_EPOCHS=50
LR=2e-05
PATIENCE=20

# #a100()
# CUDA=2
# BATCH_SIZE=16
# DEV_BATCH_SIZE=16
# TEST_BATCH_SIZE=16
# NUM_TRAIN_EPOCHS=50
# LR=2e-05
# PATIENCE=20

MODEL_NAME='codet5'
#codebert
TASK='summarize'
#summarize
SUB_TASK='javascript'
#python


TAG='examplas_training_on_disturbed'

DATA_NUM=-1
MODEL_DIR=save_models/${TAG}
SUMMARY_DIR=tensorboard/${TAG}
FULL_MODEL_TAG=${MODEL_NAME}


OUTPUT_DIR=${MODEL_DIR}/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
ORIGIN_MODEL_DIR=${ORIGIN_MODEL_DIR}/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
RES_DIR=results/${TAG}/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
RES_FN=results/${TAG}/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}.txt

CACHE_DIR=${WORKDIR}/.cache/${TAG}/${TASK}/${SUB_TASK}/${FULL_MODEL_TAG}
LOG=${OUTPUT_DIR}/train.log
mkdir -p ${OUTPUT_DIR}
mkdir -p ${CACHE_DIR}
mkdir -p ${RES_DIR}

RUN_FN=${WORKDIR}/main_examplas.py

CUDA_VISIBLE_DEVICES=${CUDA} \
TOKENIZERS_PARALLELISM=false \
python ${RUN_FN} ${MULTI_TASK_AUG} \
--do_train \
--do_eval \
--do_eval_bleu \
--do_test \
--save_last_checkpoints \
--always_save_model \
--seed 1234 \
\
--model_name ${MODEL_NAME} \
--task ${TASK} \
--sub_task ${SUB_TASK} \
--data_num ${DATA_NUM} \
--few_shot 11605 \
--output_dir ${OUTPUT_DIR} \
--summary_dir ${SUMMARY_DIR} \
--huggingface_locals ${HUGGINGFACE_LOCALS} \
--data_dir ${WORKDIR}/data \
--cache_path ${CACHE_DIR} \
--res_dir ${RES_DIR} \
--res_fn ${RES_FN} \
\
--batch_size ${BATCH_SIZE} \
--dev_batch_size ${DEV_BATCH_SIZE} \
--test_batch_size ${TEST_BATCH_SIZE} \
--num_train_epochs ${NUM_TRAIN_EPOCHS} \
--lr ${LR} \
--patience ${PATIENCE} \
--warmup_steps 1000 \
--adam_epsilon 1e-08 \
--max_grad_norm 1.0 \
--weight_decay 0.0 \
--start_epoch 0 \
--beam_size 10 \
\
--max_source_length 256 \
--max_target_length 128 \
\
--gradient_accumulation_steps 1 \
--local_rank -1 \
\
--origin_model_dir ${ORIGIN_MODEL_DIR} \
2>&1 | tee ${LOG}