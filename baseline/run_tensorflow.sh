#!/bin/bash
. ~/miniconda3/etc/profile.d/conda.sh && conda init
conda activate tensorflow
LOG_DIR=/root/mlir/baseline/_results/tensorflow
mkdir -p ${LOG_DIR}

# attention
# ATTENTION_DIR=${LOG_DIR}/Attention
# mkdir -p ${ATTENTION_DIR}

# cd attention
# nsys nvprof python3 attention_tf2.py --bs 1 > ${ATTENTION_DIR}/attention.b1.log 2>&1
#     #unroll
# nsys nvprof python3 attention_tf2.py --unroll --bs 1 > ${ATTENTION_DIR}/attention_unroll.b1.log 2>&1
# rm report*
# cd ..

# #lstm
# LSTM_DIR=${LOG_DIR}/Lstm
# mkdir -p ${LSTM_DIR}
# cd lstm
# nsys nvprof python3 lstm_tf2.py --fix --bs 1 > ${LSTM_DIR}/lstm.fix.b1.log 2>&1
# nsys nvprof python3 lstm_tf2.py --unroll --bs 1 > ${LSTM_DIR}/lstm.unroll.b1.log 2>&1
# rm report*
# cd ..

# #Nasrnn
# NASRNN_DIR=${LOG_DIR}/Nasrnn
# mkdir -p ${NASRNN_DIR}
# cd nasrnn
# nsys nvprof python3 nas_tf2.py --bs 1 > ${NASRNN_DIR}/nasrnn.fix.b1.log 2>&1
# nsys nvprof python3 nas_tf2.py --unroll --bs 1 > ${NASRNN_DIR}/nasrnn.unroll.b1.log 2>&1
# rm report*
# cd ..

# #Rae
# RAE_DIR=${LOG_DIR}/Rae
# mkdir -p ${RAE_DIR}
# cd rae
# nsys nvprof python3 rae_tf2.py> ${RAE_DIR}/rae.eager.b1.log 2>&1
# rm report*
# cd ..

# #seq2seq
# SEQ2SEQ_DIR=${LOG_DIR}/Seq2seq
# mkdir -p ${SEQ2SEQ_DIR}
# cd seq2seq
# nsys nvprof python3 seq2seq_tf2.py> ${SEQ2SEQ_DIR}/seq2seq.eager.b1.log 2>&1
# nsys nvprof python3 seq2seq_tf2.py --unroll> ${SEQ2SEQ_DIR}/seq2seq.eager.unroll.b1.log 2>&1
# rm report*
# cd ..

conda deactivate
conda activate baseline_tf1


# #blockdrop
# BLOCKDROP_DIR=${LOG_DIR}/Blockdrop
# mkdir -p ${BLOCKDROP_DIR}
# cd blockdrop
# nsys nvprof python3 blockdrop_tf.py --fix --bs 1 > ${BLOCKDROP_DIR}/blockrop.fix.rn1.b1.log 2>&1
# nsys nvprof python3 blockdrop_tf.py --fix --bs 64 > ${BLOCKDROP_DIR}/blockrop.fix.rn1.b64.log 2>&1
# nsys nvprof python3 blockdrop_tf.py --unroll --bs 1 > ${BLOCKDROP_DIR}/blockrop.unroll.rn1.b1.log 2>&1
# nsys nvprof python3 blockdrop_tf.py --unroll --bs 64 > ${BLOCKDROP_DIR}/blockrop.unroll.rn1.b64.log 2>&1

# nsys nvprof python3 blockdrop_tf.py --unroll --bs 1 --rate 0 > ${BLOCKDROP_DIR}/blockrop.fix.r0.b1.log 2>&1
# nsys nvprof python3 blockdrop_tf.py --unroll --bs 1 --rate 25 > ${BLOCKDROP_DIR}/blockrop.fix.r25.b1.log 2>&1
# nsys nvprof python3 blockdrop_tf.py --unroll --bs 1 --rate 50 > ${BLOCKDROP_DIR}/blockrop.fix.r50.b1.log 2>&1
# nsys nvprof python3 blockdrop_tf.py --unroll --bs 1 --rate 75 > ${BLOCKDROP_DIR}/blockrop.fix.r75.b1.log 2>&1
# nsys nvprof python3 blockdrop_tf.py --unroll --bs 1 --rate 100 > ${BLOCKDROP_DIR}/blockrop.fix.r100.b1.log 2>&1
# rm report*
# cd ..



#skipnet
SKIPNET_DIR=${LOG_DIR}/Skipnet
mkdir -p ${SKIPNET_DIR}
cd skipnet
nsys nvprof python3 skipnet_tf.py --fix --bs 1 > ${SKIPNET_DIR}/skipnet.fix.b1.log 2>&1
nsys nvprof python3 skipnet_tf.py --fix --bs 64 > ${SKIPNET_DIR}/skipnet.fix.b64.log 2>&1
nsys nvprof python3 skipnet_tf.py --unroll --bs 1 > ${SKIPNET_DIR}/skipnet.unroll.b1.log 2>&1
nsys nvprof python3 skipnet_tf.py --unroll --bs 64 > ${SKIPNET_DIR}/skipnet.unroll.b64.log 2>&1

nsys nvprof python3 skipnet_tf.py --unroll --bs 1 --rate 0 > ${SKIPNET_DIR}/skipnet.fix.r0.b1.log 2>&1
nsys nvprof python3 skipnet_tf.py --unroll --bs 1 --rate 25 > ${SKIPNET_DIR}/skipnet.fix.r25.b1.log 2>&1
nsys nvprof python3 skipnet_tf.py --unroll --bs 1 --rate 50 > ${SKIPNET_DIR}/skipnet.fix.r50.b1.log 2>&1
nsys nvprof python3 skipnet_tf.py --unroll --bs 1 --rate 75 > ${SKIPNET_DIR}/skipnet.fix.r75.b1.log 2>&1
nsys nvprof python3 skipnet_tf.py --unroll --bs 1 --rate 100 > ${SKIPNET_DIR}/skipnet.fix.r100.b1.log 2>&1
rm report*
cd ..
conda deactivate
