
#!/bin/bash

WorkspaceDir="/home/jykim/bfuse-project"
FusionTestDir="${WorkspaceDir}/test/fusion-test"

${WorkspaceDir}/build/bin/bfuseplus -p ${FusionTestDir} -c ${FusionTestDir}/configs -d ${FusionTestDir}/results