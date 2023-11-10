
#!/bin/bash

WorkspaceDir="/root/bfuse-project-tvm"
FusionTestDir="${WorkspaceDir}/test/fusion-test"

${WorkspaceDir}/build/bin/bfuseplus -p ${FusionTestDir} -c ${FusionTestDir}/configs -d ${FusionTestDir}/results
