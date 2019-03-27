#!/bin/bash

# -------- dat
USE_CUDA=true
DAT=true
exp=dat
# -------- config
config=config.ini
# --- training
LR_schedule=ReduceLROnPlateau
LR=0.0001
#LR_size=1
#LR_factor=0.5
MAX_ITER=300
SAVE_MODEL=true
SELECT_BEST_MODEL=false
SAVE_ITER=1
# --- debug
debug= #--debug-mode
# --- paths
#wdir=/share/spandh.ami1/emo/dev/6class/vlog/mosei/tools/audioemotion/audio-attention/
wdir=/share/mini3/mini/emo/dev/6class/vlog/mosei/tools/audioemotion/audio-attention
# --------


# submitting jobs
testall= #True
jid="-"
prev_epoch=1
python=/share/spandh.ami1/sw/std/python/anaconda3-5.1.0/v5.1.0/bin/python
submitjob=/share/spandh.ami1/sw/mini/jet/latest/tools/submitjob

#mkdir -p $wdir/killall
killscript=$wdir/killall.sh
rm -f ${killscript} # execute?
echo '#!/bin/bash' >> ${killscript}
chmod 777 ${killscript}



for traindatalbl in MOSEI_acl2018_neNA+ent05p2_t34v5t5_shoutclipped+ravdess_t17v2t5_all1neNAcaNA_shoutclipped+iemocap_t1234t5_neNAfrNAexNAotNA
 
#MOSEI_acl2018+ent05p2_t34v5t5_shoutclipped+ravdess_t18v3t3_all1ne0caNA_shoutclipped+iemocap_t1234t5_ne0frNAexNAotNA
#ent05p2_t34v5t5_shoutclipped+ravdess_t18v3t3_all1ne0caNA_shoutclipped

# MOSEI_acl2018+ent05p2_t34v5t5_shoutclipped+ravdess_t18v3t3_all1ne0caNA_shoutclipped+iemocap_t1234t5_ne0frNAexNAotNA 
do
#for testdatalbl in ent05p2_t34v5t5_shoutclipped+ravdess_t18v3t3_all1ne0caNA_shoutclipped
#do
  for ext in fbk
  do

    # change config
    if [ "$ext" == "fbk" ]; then
      inputsize=23
    elif [ "$ext" == "fbk40" ]; then
      inputsize=40
    elif [ "$ext" == "fbk60" ]; then
      inputsize=60
    elif [ "$ext" == "fbk80" ]; then
      inputsize=80
    elif [ "$ext" == "covarep" ]; then
      inputsize=74
    elif [ "$ext" == "plp" ]; then
      inputsize=14
    elif [ "$ext" == "mfcc" ]; then
      inputsize=13
    fi



    for LR_size in 4 #6 7 8 9 #`seq 1 6 10`
    do
    for LR_factor in 0.8 #0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
    do
    for BATCHSIZE in 1 #64 #32 1 #64 128 256
    do


    for prev_epoch in `seq 1 20 $MAX_ITER`
    do

	# make path
        path=train-${traindatalbl}-${ext}-$LR$LR_schedule${LR_size}h${LR_factor}
        if [ "$BATCHSIZE" != 1 ]; then
          path=$path-BS$BATCHSIZE
        fi
	if [ "$SELECT_BEST_MODEL" == true ]; then
          path=$path-SBM
	  SAVE_ITER=1
        fi
	fpath=$wdir/$exp/$path
	mkdir -p $fpath
	
	# copy completed config
        newconfig=$fpath/$config
        rm -f $newconfig
	cp $config $newconfig
        chmod u+w $newconfig

	# update config
	if [ "$debug" == "--debug-mode" ]; then
	  sed -i "s/DEBUG_MODE=false/DEBUG_MODE=true/g" $newconfig
	else
	  sed -i "s/DEBUG_MODE=true/DEBUG_MODE=false/g" $newconfig
	fi
        sed -i "s/EXT=fbk/EXT=$ext/g" $newconfig
	sed -i "s/path=none/path=${path}/g" $newconfig
	sed -i "s/input_size=23/input_size=$inputsize/g" $newconfig
	sed -i "s/MAX_ITER=100/MAX_ITER=$MAX_ITER/g" $newconfig
	sed -i "s/LEARNING_RATE=0.0001/LEARNING_RATE=$LR/g" $newconfig
	sed -i "s/LR_schedule=ReduceLROnPlateau/LR_schedule=${LR_schedule}/g" $newconfig
        sed -i "s/LR_size=10/LR_size=${LR_size}/g" $newconfig
        sed -i "s/LR_factor=0.1/LR_factor=${LR_factor}/g" $newconfig
        sed -i "s/BATCHSIZE=1/BATCHSIZE=${BATCHSIZE}/g" $newconfig
        sed -i "s/SAVE_MODEL=true/SAVE_MODEL=${SAVE_MODEL}/g" $newconfig
        sed -i "s/SAVE_ITER=5/SAVE_ITER=${SAVE_ITER}/g" $newconfig
	sed -i "s/SELECT_BEST_MODEL=false/SELECT_BEST_MODEL=${SELECT_BEST_MODEL}/g" $newconfig
        sed -i "s/USE_CUDA=true/USE_CUDA=${USE_CUDA}/g" $newconfig

	sed -i "s/exp=none/exp=${exp}/g" $newconfig
	sed -i "s/DAT=false/DAT=true/g" $newconfig
#	sed -i "s/num_domains=4/num_domains=2/g" $newconfig


	# make scripts
	fpath2=$fpath/train${prev_epoch}-${MAX_ITER}${debug}
        L=${fpath2}.log
        S=${fpath2}.sh


        # create bash script
        echo "#!/usr/bin/bash" > $S
        echo "$python $wdir/attention_model_dat.py --train $traindatalbl --test $testdatalbl --train-mode --epochs $prev_epoch $MAX_ITER -c $newconfig" >> $S
        chmod u+x $S


        if [ "$USE_CUDA" == "false" ]; then
	    qtype=NORMAL
        else
	    qtype=GPU
	fi

        # submitjob
        if [ "$jid" == "-" ]; then
          jid=`$submitjob -q $qtype -p MINI -o -l hostname="node20|node21|node22|node23|node24|node25|node26" -eo  $L $S | tail -1`
        else
          jid=`$submitjob -q $qtype -p MINI -o -l hostname="node20|node21|node22|node23|node24|node25|node26" -eo -w $jid $L $S | tail -1`
        fi
        echo "$S $L $jid"
        echo "qdel $jid" >> ${killscript}

	if [ "$debug" == "--debug-mode" ]; then
          exit
	fi

#	exit

      done

      # testing all data
      if [ "$testall" == "True" ]; then
        # test
        testdatalbl=MOSEI_acl2018+ent05p2_t34v5t5_shoutclipped+ravdess_t18v3t3_all1ne0caNA_shoutclipped+iemocap_t1234t5_ne0frNAexNAotNA+iemocap_t1234t5_haex1sa1an1ne0
        scriptbase=$fpath/test${testdatalbl}${debug}
        L=${scriptbase}.log
        S=${scriptbase}.sh
        # create bash script
        echo "#!/usr/bin/bash" > $S
        echo "$python $wdir/test_attention.py -e $ext --train $traindatalbl --test $testdatalbl $debug --epochs 300 300" >> $S
        chmod u+x $S

        # submitjob
        jid=`$submitjob -q GPU -p MINI -o -l hostname="node20|node21|node22|node23|node24|node25|node26" -eo -w $jid $L $S | tail -1`
        echo "$S $L $jid"
        echo "qdel $jid" >> ${killscript}
      fi

      # reset
      jid="-"

done
done
done
done
done
#done

