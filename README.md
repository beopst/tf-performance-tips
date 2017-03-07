# Optimizing TensorFlow code

This repo summarizes some techniques for optimizing TensorFlow code.Official document describing a collection of best practices can be found [here](https://www.tensorflow.org/performance/performance_guide). Before started, it will very helpful to read that document.

Dockerfile which contains all of packages introduced in this document is provided. This file includes how to install the libraries/packages listed below.

First of all, it is important to find whether CPU will bottleneck GPU, or vice versa (simply check by running `nvidia-smi`). If GPU is a bottleneck, it is relatively easy to optimize. On the other hand, it is complicated if CPU is your bottleneck.

Overall, I got 1.5~2.0x performance gain by applying all belows.

## If GPUs are fully utilized

1. Use `NCHW` data format for 4D tensor.
  * Native data format for cudnn library is `NCHW`. Performance gain increases as you have many layers.
  * If you use this format, using `_fused_batch_norm` is mandatory. Otherwise, your code will be almost 10x slower since `nn.moments` cannot deal with this format effciently.
  * Several preprocessing ops support `CHW` format, so we have to transpose tensors somewhere. If your input pipeline is a bottleneck, it is better to transpose them using GPU.
2. Use fused batch norm.
  * Whatever your data format is, it is better to use fused batch norm.

## If CPUs are your bottleneck

1. Utilize queues for input pipieline
  * First, you have to utilize queues for reading and fetching input data. Please refer to [Reading Data Guide](https://www.tensorflow.org/programmers_guide/reading_data#reading_from_files) and [`batch_inputs` function](https://github.com/tensorflow/models/blob/master/inception/inception/image_processing.py#L407) in inception codes.
  * CAREFULLY allocate threads for each reading and preprocessing.
2. Use TCMalloc.
  * [TCMalloc](http://goog-perftools.sourceforge.net/doc/tcmalloc.html) is faster for multi-threaded programs.
  * Also, it is effective if you use multi-threads for input pipeline.
  * Relevant issues or comments: [here](https://github.com/tensorflow/tensorflow/issues/3009#issuecomment-235993119), [here](https://github.com/tensorflow/tensorflow/issues/6779).
3. Use advanced instructions (SSE, AVX, FMA) on Intel CPUs.
  * For TensorFlow v.1.0.0, you can see the following warnings when you execute codes.
  ```
  tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
  tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
  tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
  tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
  tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
  tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use FMA instructions, but these are available on your machine and could speed up CPU computations.
  ```
  * To use these instructions, you have to build from sources. Simple way is to build dockerfile.
  * Relevant issues or comments: [here](https://github.com/tensorflow/tensorflow/issues/7449), [here](https://github.com/tensorflow/tensorflow/issues/7778), [here](https://github.com/tensorflow/tensorflow/issues/7693).
