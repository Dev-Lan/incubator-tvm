<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

Install
-------

Follow instructions on https://docs.tvm.ai/install/from_source.html, 
including LLVM, Python Method 1 installation, and all Python dependencies
such as ANTLR4.

Read Data
---------

For reading data,
```
import numpy as np
data = np.load('task%i.npz' % i)
x_train_configs = data['x_train_configs']
x_train = data['x_train']
y_train = data['y_train']
x_test_configs = data['x_test_configs']
x_test = data['x_test']
y_test = data['y_test']
task = data['task'][0]
```

Or for reading original unflattened data (requires TVM install):
```
import pickle
with open('task0.pkl', 'rb') as f:
    features = pickle.load(f)
```
Data is contained in `features[index].config`, `features[index].feature`, and `features[index].result`.



<img src=https://raw.githubusercontent.com/apache/incubator-tvm-site/master/images/logo/tvm-logo-small.png width=128/> Open Deep Learning Compiler Stack
==============================================
[Documentation](https://docs.tvm.ai) |
[Contributors](CONTRIBUTORS.md) |
[Community](https://tvm.apache.org/community) |
[Release Notes](NEWS.md)

[![Build Status](https://ci.tvm.ai/buildStatus/icon?job=tvm/master)](https://ci.tvm.ai/job/tvm/job/master/)
[![Azure Pipeline](https://dev.azure.com/tvmai/tvm/_apis/build/status/windows_mac_build?branchName=master)](https://dev.azure.com/tvmai/tvm/_build/latest?definitionId=2&branchName=master)

Apache TVM (incubating) is a compiler stack for deep learning systems. It is designed to close the gap between the
productivity-focused deep learning frameworks, and the performance- and efficiency-focused hardware backends.
TVM works with deep learning frameworks to provide end to end compilation to different backends.

License
-------
© Contributors Licensed under an [Apache-2.0](LICENSE) license.

Contribute to TVM
-----------------
TVM adopts apache committer model, we aim to create an open source project that is maintained and owned by the community.
Checkout the [Contributor Guide](https://docs.tvm.ai/contribute/)

Acknowledgement
---------------
We learned a lot from the following projects when building TVM.
- [Halide](https://github.com/halide/Halide): TVM uses [HalideIR](https://github.com/dmlc/HalideIR) as data structure for
  arithmetic simplification and low level lowering. We also learned and adapted some part of lowering pipeline from Halide.
- [Loopy](https://github.com/inducer/loopy): use of integer set analysis and its loop transformation primitives.
- [Theano](https://github.com/Theano/Theano): the design inspiration of symbolic scan operator for recurrence.
