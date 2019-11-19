// Copyright 2017-2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// This script acts as a trigger script for the main he-transformer-unittest.groovy
// Jenkins job.  This script is part of a Jenkins multi-branch pipeline job
// which can trigger GitHub jobs more effectively than the GitHub Pull
// Request Builder (GHPRB) plugin, in our environment.

// The original he-transformer-unittest job required the following parameters.  We
// set these up below as global variables, so we do not need to rewrite the
// original script -- we only need to provide this new trigger hook.
//

String JENKINS_BRANCH = "master"
String TIMEOUTTIME = "3600"

// Constants
String JENKINS_DIR = "jenkins"

timestamps {

    node("trigger") {

        deleteDir()  // Clear the workspace before starting

        // Clone the cje-algo directory which contains our Jenkins groovy scripts
        def sleeptime=0
        retry(count: 3) {
            sleep sleeptime; sleeptime = 10
            sh "git clone -b $JENKINS_BRANCH https://github.intel.com/AIPG/cje-algo $JENKINS_DIR"
        }

        def heTransformerCIPreMerge = load("$JENKINS_DIR/hetransformer-lib/he-transformer-ci-premerge.groovy")
        heTransformerCIPreMerge(prURL: CHANGE_URL,
                        prAuthor: CHANGE_AUTHOR,
                        useMBPipelineSCM: 'true',
                        checkoutBranch: '-UNDEFINED-BRANCH-'
                        )

        echo "he-transformer-ci-premerge.groovy completed"

    }  // End:  node

}  // End:  timestamps

echo "Done"
