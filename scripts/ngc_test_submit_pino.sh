ngc batch run --name 'ml-model.PINO.ns-dat800' --preempt RESUMABLE \
--commandline 'cd /Code/PINO; git pull; bash scripts/train_dat800.sh' \
--image 'nvidia/pytorch:22.08-py3' \
--priority HIGH \
--ace nv-us-west-2 \
--instance dgxa100.40g.1.norm \
--workspace QsixjfOES8uYIp5kwIDblQ:/Code \
--datasetid 111345:/mount/data \
--team nvr-aialgo \
--result /results