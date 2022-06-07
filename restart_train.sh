#!/bin/bash
#!
#! Example SLURM job script for Wilkes3 (AMD EPYC 7763, ConnectX-6, A100)
#! Last updated: Fri 30 Jul 11:07:58 BST 2021
#!

#!#############################################################
#!#### Modify the options in this section as appropriate ######
#!#############################################################

#! sbatch directives begin here ###############################
#! Name of the job:
#SBATCH -J d3st_v35
#! Which project should be charged (NB Wilkes2 projects end in '-GPU'):
#SBATCH -A GASIC-BHT26-SL2-GPU
#! How many whole nodes should be allocated?
#SBATCH --nodes=1
#! How many (MPI) tasks will there be in total?
#! Note probably this should not exceed the total number of GPUs in use.
#SBATCH --ntasks=1
#! Specify the number of GPUs per node (between 1 and 4; must be 4 if nodes>1).
#! Note that the job submission script will enforce no more than 32 cpus per GPU.
#SBATCH --gres=gpu:1
#! How much wallclock time will be required?
#SBATCH --time=36:00:00
#! What types of email messages do you wish to receive?
#SBATCH --mail-type=NONE
#! Uncomment this to prevent the job from being requeued (e.g. if
#! interrupted by node failure or system downtime):
##SBATCH --no-requeue

#! Do not change:
#SBATCH -p ampere

#! sbatch directives end here (put any additional directives above this line)

#! Notes:
#! Charging is determined by GPU number*walltime.

#! Number of nodes and tasks per node allocated by SLURM (do not change):
numnodes=$SLURM_JOB_NUM_NODES
numtasks=$SLURM_NTASKS
mpi_tasks_per_node=$(echo "$SLURM_TASKS_PER_NODE" | sed -e  's/^\([0-9][0-9]*\).*$/\1/')
#! ############################################################
#! Modify the settings below to specify the application's environment, location
#! and launch method:

#! Optionally modify the environment seen by the application
#! (note that SLURM reproduces the environment at submission irrespective of ~/.bashrc):
. /etc/profile.d/modules.sh                # Leave this line (enables the module command)
module purge                               # Removes all modules still loaded
module load rhel8/default-amp              # REQUIRED - loads the basic environment


if [ -z ${CONDA_ENV_PATH+x} ]; then
  echo "Please pass the absolute path to your conda environment by prepending CONDA_ENV_PATH=abs/path/to/the/args variable."
  exit
fi

#! Insert additional module load commands after this line if needed:
module load python/3.8
module load miniconda/3
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_PATH"
which python

if [ -z ${CRS+x} ]; then
  echo "Please your CRS. For example prepend CRS=ac2123"
  exit
fi

if [ -z ${$AUG_DATA_DIR+x} ]; then
  echo "Please pass the name of the directory where the augmented data directory is located. This should be a directory
  located in */d3st/data/processed/. It should contain v1-v5 directories as well as the original/ directory."
  exit
fi


if [ -z ${VERSION+x} ]; then
  echo "Please pass an integer representing the data version to the command. For example prepend VERSION=1 to
  test data preprocessed in folder */version_1/"
  exit
fi

if [ -z ${ARGS_REL_PATH+x} ]; then
  echo "Please specify the path of the training arguments file relative to configs/ dir. For example, ARGS_REL_PATH=main_experiments/training/hpc_train_arguments.yaml."
  exit
fi

if [ -z ${EXPERIMENT+x} ]; then
  echo "Please specify the experiment name from which you would like to continue by prepending EXPERIMENT=[name] where [name] is a
  placeholder for your experiment name"
  exit
fi


#! Full path to application executable:
application="python -m scripts.train"

#! Run options for the application:
options="-s /home/$CRS/rds/rds-wjb31-nmt2020/ac2123/d3st/data/raw/sgd/train/schema.json \
-d /home/$CRS/rds/rds-wjb31-nmt2020/ac2123/d3st/data/preprocessed/original/dev/version_$VERSION/data.json \
-t /home/$CRS/rds/rds-wjb31-nmt2020/ac2123/d3st/data/preprocessed/$AUG_DATA_DIR/original/train/version_$VERSION/data.json \
-t /home/$CRS/rds/rds-wjb31-nmt2020/ac2123/d3st/data/preprocessed/$AUG_DATA_DIR/v1/train/version_$VERSION/data.json \
--ref_dir /home/$CRS/rds/rds-wjb31-nmt2020/ac2123/dstc8-schema-guided-dialogue/sgd_x/data/raw/original/dev
--template_dir /home/$CRS/rds/rds-wjb31-nmt2020/ac2123/d3st/data/interim/blank_dialogue_templates/original/dev
--restore /home/$CRS/rds/rds-wjb31-nmt2020/$CRS/d3st/models/$EXPERIMENT/version_$VERSION/model.last
--do_inference
--override
-a configs/$ARGS_REL_PATH -vvv"

#! Work directory (i.e. where the job will run):
workdir="$SLURM_SUBMIT_DIR"  # The value of SLURM_SUBMIT_DIR sets workdir to the directory
                             # in which sbatch is run.

#! Are you using OpenMP (NB this is unrelated to OpenMPI)? If so increase this
#! safe value to no more than 128:
export OMP_NUM_THREADS=1

#! Number of MPI tasks to be started by the application per node and in total (do not change):
np=$[${numnodes}*${mpi_tasks_per_node}]

#! Choose this for a pure shared-memory OpenMP parallel program on a single node:
#! (OMP_NUM_THREADS threads will be created):
CMD="$application $options"

#! Choose this for a MPI code using OpenMPI:
#CMD="mpirun -npernode $mpi_tasks_per_node -np $np $application $options"


###############################################################
### You should not have to change anything below this line ####
###############################################################

cd $workdir
echo -e "Changed directory to `pwd`.\n"

JOBID=$SLURM_JOB_ID

echo -e "JobID: $JOBID\n======"
echo "Time: `date`"
echo "Running on master node: `hostname`"
echo "Current directory: `pwd`"

if [ "$SLURM_JOB_NODELIST" ]; then
        #! Create a machine file:
        export NODEFILE=`generate_pbs_nodefile`
        cat $NODEFILE | uniq > machine.file.$JOBID
        echo -e "\nNodes allocated:\n================"
        echo `cat machine.file.$JOBID | sed -e 's/\..*$//g'`
fi

echo -e "\nnumtasks=$numtasks, numnodes=$numnodes, mpi_tasks_per_node=$mpi_tasks_per_node (OMP_NUM_THREADS=$OMP_NUM_THREADS)"

echo -e "\nExecuting command:\n==================\n$CMD\n"

eval $CMD
