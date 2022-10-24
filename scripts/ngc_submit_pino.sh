ngc batch run --name 'ml-model.PINO.ns-dat400' --preempt RESUMABLE \
--commandline 'cd /Code/PINO; git pull; bash scripts/train_dat400.sh' \
--image 'nvidia/pytorch:22.08-py3' \
--priority HIGH \
--ace nv-us-west-2 \
--instance dgxa100.40g.1.norm \
--workspace QsixjfOES8uYIp5kwIDblQ:/Code \
--datasetid 111345:/mount/data \
--team nvr-aialgo \
--result /results

ngc batch run --name 'ml-model.PINO.ns-dat200' --preempt RESUMABLE \
--commandline 'cd /Code/PINO; git pull; bash scripts/train_dat200.sh' \
--image 'nvidia/pytorch:22.08-py3' \
--priority HIGH \
--ace nv-us-west-2 \
--instance dgxa100.40g.1.norm \
--workspace QsixjfOES8uYIp5kwIDblQ:/Code \
--datasetid 111345:/mount/data \
--team nvr-aialgo \
--result /results

ngc batch run --name 'ml-model.PINO.ns-dat80' --preempt RESUMABLE \
--commandline 'cd /Code/PINO; git pull; bash scripts/train_dat80.sh' \
--image 'nvidia/pytorch:22.08-py3' \
--priority HIGH \
--ace nv-us-west-2 \
--instance dgxa100.40g.1.norm \
--workspace QsixjfOES8uYIp5kwIDblQ:/Code \
--datasetid 111345:/mount/data \
--team nvr-aialgo \
--result /results

ngc batch run --name 'ml-model.PINO.ns-dat40' --preempt RESUMABLE \
--commandline 'cd /Code/PINO; git pull; bash scripts/train_dat40.sh' \
--image 'nvidia/pytorch:22.08-py3' \
--priority HIGH \
--ace nv-us-west-2 \
--instance dgxa100.40g.1.norm \
--workspace QsixjfOES8uYIp5kwIDblQ:/Code \
--datasetid 111345:/mount/data \
--team nvr-aialgo \
--result /results

ngc batch run --name 'ml-model.PINO.ns-dat0' --preempt RESUMABLE \
--commandline 'cd /Code/PINO; git pull; bash scripts/train_dat0.sh' \
--image 'nvidia/pytorch:22.08-py3' \
--priority HIGH \
--ace nv-us-west-2 \
--instance dgxa100.40g.1.norm \
--workspace QsixjfOES8uYIp5kwIDblQ:/Code \
--datasetid 111345:/mount/data \
--team nvr-aialgo \
--result /results

ngc batch run --name 'ml-model.PINO.ns-res32' --preempt RESUMABLE \
--commandline 'cd /Code/PINO; git pull; bash scripts/train_res32.sh' \
--image 'nvidia/pytorch:22.08-py3' \
--priority HIGH \
--ace nv-us-west-2 \
--instance dgxa100.40g.1.norm \
--workspace QsixjfOES8uYIp5kwIDblQ:/Code \
--datasetid 111345:/mount/data \
--team nvr-aialgo \
--result /results

ngc batch run --name 'ml-model.PINO.ns-res16' --preempt RESUMABLE \
--commandline 'cd /Code/PINO; git pull; bash scripts/train_res16.sh' \
--image 'nvidia/pytorch:22.08-py3' \
--priority HIGH \
--ace nv-us-west-2 \
--instance dgxa100.40g.1.norm \
--workspace QsixjfOES8uYIp5kwIDblQ:/Code \
--datasetid 111345:/mount/data \
--team nvr-aialgo \
--result /results