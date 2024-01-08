#!/bin/bash
. ~/miniconda3/etc/profile.d/conda.sh && conda init
conda activate pytorch
LOG_DIR=/root/mlir/baseline/_results/pytorch
mkdir -p ${LOG_DIR}

#attention
ATTENTION_DIR=${LOG_DIR}/Attention
mkdir -p ${ATTENTION_DIR}

cd attention
nsys nvprof python3 attention_pytorch.py --bs 1 > ${ATTENTION_DIR}/attention.b1.log 2>&1
nsys nvprof python3 attention_pytorch.py --bs 64 > ${ATTENTION_DIR}/attention.b64.log 2>&1
    #unroll
nsys nvprof python3 attention_pytorch.py --unroll --bs 1 > ${ATTENTION_DIR}/attention_unroll.b1.log 2>&1
nsys nvprof python3 attention_pytorch.py --bs 64 > ${ATTENTION_DIR}/attention_unroll.b64.log 2>&1
rm report*
cd ..

#blockdrop
BLOCKDROP_DIR=${LOG_DIR}/Blockdrop
mkdir -p ${BLOCKDROP_DIR}
cd blockdrop
nsys nvprof python3 blockdrop_pytorch.py --fix --bs 1 > ${BLOCKDROP_DIR}/blockrop.fix.rn1.b1.log 2>&1
nsys nvprof python3 blockdrop_pytorch.py --fix --bs 64 > ${BLOCKDROP_DIR}/blockrop.fix.rn1.b64.log 2>&1
nsys nvprof python3 blockdrop_pytorch.py --unroll --bs 1 > ${BLOCKDROP_DIR}/blockrop.unroll.rn1.b1.log 2>&1
nsys nvprof python3 blockdrop_pytorch.py --unroll --bs 64 > ${BLOCKDROP_DIR}/blockrop.unroll.rn1.b64.log 2>&1

nsys nvprof python3 blockdrop_pytorch.py --unroll --bs 1 --rate 0 > ${BLOCKDROP_DIR}/blockrop.fix.r0.b1.log 2>&1
nsys nvprof python3 blockdrop_pytorch.py --unroll --bs 1 --rate 25 > ${BLOCKDROP_DIR}/blockrop.fix.r25.b1.log 2>&1
nsys nvprof python3 blockdrop_pytorch.py --unroll --bs 1 --rate 50 > ${BLOCKDROP_DIR}/blockrop.fix.r50.b1.log 2>&1
nsys nvprof python3 blockdrop_pytorch.py --unroll --bs 1 --rate 75 > ${BLOCKDROP_DIR}/blockrop.fix.r75.b1.log 2>&1
nsys nvprof python3 blockdrop_pytorch.py --unroll --bs 1 --rate 100 > ${BLOCKDROP_DIR}/blockrop.fix.r100.b1.log 2>&1
rm report*
cd ..

#lstm
LSTM_DIR=${LOG_DIR}/Lstm
mkdir -p ${LSTM_DIR}
cd lstm
nsys nvprof python3 lstm_pytorch.py --fix --bs 1 > ${LSTM_DIR}/lstm.fix.b1.log 2>&1
nsys nvprof python3 lstm_pytorch.py --fix --bs 64 > ${LSTM_DIR}/lstm.fix.b64.log 2>&1
nsys nvprof python3 lstm_pytorch.py --unroll --bs 1 > ${LSTM_DIR}/lstm.unroll.b1.log 2>&1
nsys nvprof python3 lstm_pytorch.py --unroll --bs 64 > ${LSTM_DIR}/lstm.unroll.b64.log 2>&1
rm report*
cd ..

#Nasrnn
NASRNN_DIR=${LOG_DIR}/Nasrnn
mkdir -p ${NASRNN_DIR}
cd nasrnn
nsys nvprof python3 nas_pytorch.py --fix --bs 1 > ${NASRNN_DIR}/nasrnn.fix.b1.log 2>&1
nsys nvprof python3 nas_pytorch.py --fix --bs 64 > ${NASRNN_DIR}/nasrnn.fix.b64.log 2>&1
nsys nvprof python3 nas_pytorch.py --unroll --bs 1 > ${NASRNN_DIR}/nasrnn.unroll.b1.log 2>&1
nsys nvprof python3 nas_pytorch.py --unroll --bs 64 > ${NASRNN_DIR}/nasrnn.unroll.b64.log 2>&1
rm report*
cd ..

#Rae
RAE_DIR=${LOG_DIR}/Rae
mkdir -p ${RAE_DIR}
cd rae
nsys nvprof python3 rae_pytorch.py --fix --bs 1 > ${RAE_DIR}/rae.fix.b1.log 2>&1
nsys nvprof python3 rae_pytorch.py --fix --bs 64 > ${RAE_DIR}/rae.fix.b64.log 2>&1
nsys nvprof python3 rae_pytorch.py --unroll --bs 1 > ${RAE_DIR}/rae.unroll.b1.log 2>&1
nsys nvprof python3 rae_pytorch.py --unroll --bs 64 > ${RAE_DIR}/rae.unroll.b64.log 2>&1
rm report*
cd ..

#seq2seq
SEQ_DIR=${LOG_DIR}/Seq2seq
mkdir -p ${SEQ_DIR}
cd seq2seq
nsys nvprof python3 seq2seq_pytorch.py --fix --bs 1 > ${SEQ_DIR}/seq2seq.fix.b1.log 2>&1
nsys nvprof python3 seq2seq_pytorch.py --fix --bs 64 > ${SEQ_DIR}/seq2seq.fix.b64.log 2>&1
nsys nvprof python3 seq2seq_pytorch.py --unroll --bs 1 > ${SEQ_DIR}/seq2seq.unroll.b1.log 2>&1
nsys nvprof python3 seq2seq_pytorch.py --unroll --bs 64 > ${SEQ_DIR}/seq2seq.unroll.b64.log 2>&1
rm report*
cd ..

#skipnet
SKIPNET_DIR=${LOG_DIR}/Skipnet
mkdir -p ${SKIPNET_DIR}
cd skipnet
nsys nvprof python3 skipnet_pytorch.py --fix --bs 1 > ${SKIPNET_DIR}/skipnet.fix.b1.log 2>&1
nsys nvprof python3 skipnet_pytorch.py --fix --bs 64 > ${SKIPNET_DIR}/skipnet.fix.b64.log 2>&1
nsys nvprof python3 skipnet_pytorch.py --unroll --bs 1 > ${SKIPNET_DIR}/skipnet.unroll.b1.log 2>&1
nsys nvprof python3 skipnet_pytorch.py --unroll --bs 64 > ${SKIPNET_DIR}/skipnet.unroll.b64.log 2>&1

nsys nvprof python3 skipnet_pytorch.py --unroll --bs 1 --rate 0 > ${SKIPNET_DIR}/skipnet.fix.r0.b1.log 2>&1
nsys nvprof python3 skipnet_pytorch.py --unroll --bs 1 --rate 25 > ${SKIPNET_DIR}/skipnet.fix.r25.b1.log 2>&1
nsys nvprof python3 skipnet_pytorch.py --unroll --bs 1 --rate 50 > ${SKIPNET_DIR}/skipnet.fix.r50.b1.log 2>&1
nsys nvprof python3 skipnet_pytorch.py --unroll --bs 1 --rate 75 > ${SKIPNET_DIR}/skipnet.fix.r75.b1.log 2>&1
nsys nvprof python3 skipnet_pytorch.py --unroll --bs 1 --rate 100 > ${SKIPNET_DIR}/skipnet.fix.r100.b1.log 2>&1
rm report*
cd ..
conda deactivate
