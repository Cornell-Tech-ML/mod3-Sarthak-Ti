# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html

# parallel numba diagnostics
I included the output of the parallel check function as specified in the instructions. Note that it talks about this allocation hoisting, but it is incorect, each thread needs its own buffer. I also verified that there are no actual race conditions by pausing one of the values and simply ahving it check the buffer which is unmodified by the others, so each core gets its own buffer. I asked in ed discussion about this allocation hoisting and was told it is not an issue/is ok.
```console
MAP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map,
/data1/lesliec/sarthak/mod3-Sarthak-Ti/minitorch/fast_ops.py (164)
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /data1/lesliec/sarthak/mod3-Sarthak-Ti/minitorch/fast_ops.py (164)
--------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                                                             |
        out: Storage,                                                                                                     |
        out_shape: Shape,                                                                                                 |
        out_strides: Strides,                                                                                             |
        in_storage: Storage,                                                                                              |
        in_shape: Shape,                                                                                                  |
        in_strides: Strides,                                                                                              |
    ) -> None:                                                                                                            |
        # TODO: Implement for Task 3.1.                                                                                   |
        # raise NotImplementedError("Need to implement for Task 3.1")                                                     |
        # Ensure that in_shape is smaller than or equal to out_shape dimensions                                           |
        assert len(in_shape) <= len(                                                                                      |
            out_shape                                                                                                     |
        ), "in_shape must be smaller than out_shape"                                                                      |
        strides_aligned = np.array_equal(out_strides, in_strides) and np.array_equal(                                     |
            out_shape, in_shape                                                                                           |
        )                                                                                                                 |
                                                                                                                          |
        # the first key idea is we check if the strides and the shapes are the same, if they are we can avoid indexing    |
        # then we use prange to parallelize the loop                                                                      |
        # regardless, the idea is find the index of the input, apply function and store in output                         |
                                                                                                                          |
        # Iterate over each element in `out` in parallel                                                                  |
        # print(out.size)                                                                                                 |
        if strides_aligned:                                                                                               |
            # If strides are aligned, we can avoid indexing, simply apply the function                                    |
            for i in prange(out.size):------------------------------------------------------------------------------------| #0
                out[i] = fn(in_storage[i])                                                                                |
        else:                                                                                                             |
            for i in prange(out.size):  # using prange for parallel looping-----------------------------------------------| #1
                out_index = np.empty(len(out_shape), dtype=np.int32)                                                      |
                in_index = np.empty(len(in_shape), dtype=np.int32)                                                        |
                # Get the multi-dimensional index for the current flat index `i`                                          |
                to_index(i, out_shape, out_index)                                                                         |
                                                                                                                          |
                # Broadcast out_index to in_index based on the input shape                                                |
                broadcast_index(out_index, out_shape, in_shape, in_index)                                                 |
                                                                                                                          |
                # Calculate flat positions in input and output based on strides                                           |
                in_position = index_to_position(in_index, in_strides)                                                     |
                out_position = index_to_position(out_index, out_strides)                                                  |
                                                                                                                          |
                # Apply the function and store the result in the output storage                                           |
                out[out_position] = fn(in_storage[in_position])                                                           |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #0, #1).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/data1/lesliec/sarthak/mod3-Sarthak-Ti/minitorch/fast_ops.py (194) is hoisted
out of the parallel loop labelled #1 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/data1/lesliec/sarthak/mod3-Sarthak-Ti/minitorch/fast_ops.py (195) is hoisted
out of the parallel loop labelled #1 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: in_index = np.empty(len(in_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip,
/data1/lesliec/sarthak/mod3-Sarthak-Ti/minitorch/fast_ops.py (235)
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /data1/lesliec/sarthak/mod3-Sarthak-Ti/minitorch/fast_ops.py (235)
---------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _zip(                                                                                                              |
        out: Storage,                                                                                                      |
        out_shape: Shape,                                                                                                  |
        out_strides: Strides,                                                                                              |
        a_storage: Storage,                                                                                                |
        a_shape: Shape,                                                                                                    |
        a_strides: Strides,                                                                                                |
        b_storage: Storage,                                                                                                |
        b_shape: Shape,                                                                                                    |
        b_strides: Strides,                                                                                                |
    ) -> None:                                                                                                             |
        # TODO: Implement for Task 3.1.                                                                                    |
        # raise NotImplementedError("Need to implement for Task 3.1")                                                      |
        strides_aligned = (                                                                                                |
            np.array_equal(b_strides, a_strides)                                                                           |
            and np.array_equal(a_shape, b_shape)                                                                           |
            and np.array_equal(a_shape, out_shape)                                                                         |
            and np.array_equal(out_shape, b_shape)                                                                         |
            and np.array_equal(out_strides, a_strides)                                                                     |
            and np.array_equal(out_strides, b_strides)                                                                     |
        )                                                                                                                  |
                                                                                                                           |
        # exact same idea as tensor_map, we check if strides and shapes are the same, if they are we can avoid indexing    |
        # and the only difference is have an a and b not just an a, so check all of those                                  |
        # Then simply find a and b index, apply function and store in output                                               |
                                                                                                                           |
        if strides_aligned:                                                                                                |
            # If strides are aligned, we can avoid indexing                                                                |
            for i in prange(out.size):-------------------------------------------------------------------------------------| #2
                out[i] = fn(a_storage[i], b_storage[i])                                                                    |
        else:                                                                                                              |
            for i in prange(out.size):-------------------------------------------------------------------------------------| #3
                out_index = np.empty(len(out_shape), dtype=np.int32)                                                       |
                a_index = np.empty(len(a_shape), dtype=np.int32)                                                           |
                b_index = np.empty(len(b_shape), dtype=np.int32)                                                           |
                # Get the multi-dimensional index for the current flat index `i`                                           |
                to_index(i, out_shape, out_index)                                                                          |
                                                                                                                           |
                # Broadcast out_index to a_index and b_index based on their respective shapes                              |
                broadcast_index(out_index, out_shape, a_shape, a_index)                                                    |
                broadcast_index(out_index, out_shape, b_shape, b_index)                                                    |
                                                                                                                           |
                # Calculate flat positions in a, b, and output based on strides                                            |
                a_position = index_to_position(a_index, a_strides)                                                         |
                b_position = index_to_position(b_index, b_strides)                                                         |
                out_position = index_to_position(out_index, out_strides)                                                   |
                                                                                                                           |
                # Apply the function element-wise on both input tensors and assign to output                               |
                out[out_position] = fn(a_storage[a_position], b_storage[b_position])                                       |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 2 parallel for-
loop(s) (originating from loops labelled: #2, #3).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/data1/lesliec/sarthak/mod3-Sarthak-Ti/minitorch/fast_ops.py (267) is hoisted
out of the parallel loop labelled #3 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.empty(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/data1/lesliec/sarthak/mod3-Sarthak-Ti/minitorch/fast_ops.py (268) is hoisted
out of the parallel loop labelled #3 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: a_index = np.empty(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at
/data1/lesliec/sarthak/mod3-Sarthak-Ti/minitorch/fast_ops.py (269) is hoisted
out of the parallel loop labelled #3 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: b_index = np.empty(len(b_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE

================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce,
/data1/lesliec/sarthak/mod3-Sarthak-Ti/minitorch/fast_ops.py (309)
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /data1/lesliec/sarthak/mod3-Sarthak-Ti/minitorch/fast_ops.py (309)
-----------------------------------------------------------------------------------------------------------------------------------|loop #ID
    def _reduce(                                                                                                                   |
        out: Storage,                                                                                                              |
        out_shape: Shape,                                                                                                          |
        out_strides: Strides,                                                                                                      |
        a_storage: Storage,                                                                                                        |
        a_shape: Shape,                                                                                                            |
        a_strides: Strides,                                                                                                        |
        reduce_dim: int,                                                                                                           |
    ) -> None:                                                                                                                     |
        # a bit more complex than map and zip, but we can still parallelize it for each output                                     |
        # this means if do .reduce() without specifying a dimension it won't actually be parallel                                  |
        # but if we have a 100x100 tensor and do .sum(0) it will parallelize across up to 100 cpu cores!                           |
        # the basic ideas is we find the index of the output, then one core will fully compute that output and store it            |
        # so in the 100x100 example, a core would take the first row, sum it and store it. each core does its own row              |
                                                                                                                                   |
        reduce_size = a_shape[reduce_dim]  # size of the dimension we are reducing                                                 |
                                                                                                                                   |
        assert len(out_shape) < MAX_DIMS, "out_shape must be less than MAX_DIMS"                                                   |
                                                                                                                                   |
        for i in prange(-----------------------------------------------------------------------------------------------------------| #4
            len(out)                                                                                                               |
        ):  # Parallelized loop, if output is 1 dimension doesn't actually parallelize, but if reduce along a dimension it will    |
            out_index = np.empty(MAX_DIMS, dtype=np.int32)  # Index buffer for output                                              |
            to_index(i, out_shape, out_index)  # get the index                                                                     |
            base_position = index_to_position(                                                                                     |
                out_index, out_strides                                                                                             |
            )  # get the position of the value that will remain after reduction                                                    |
                                                                                                                                   |
            # Inner loop: use precomputed offsets for reduction                                                                    |
            o = index_to_position(out_index, out_strides)                                                                          |
            result = out[o]  # Identity for the reduction, 0 for sum, 1 for product                                                |
            out_index[reduce_dim] = 0  # initialize to 0                                                                           |
            temp_position = index_to_position(                                                                                     |
                out_index, a_strides                                                                                               |
            )  # the starting position of the values to reduce                                                                     |
            for s in range(reduce_size):                                                                                           |
                result = fn(                                                                                                       |
                    result, float(a_storage[temp_position + s * a_strides[reduce_dim]])                                            |
                )  # apply the function to the result and the next value and keep applying it until it's done                      |
                                                                                                                                   |
            out[base_position] = result                                                                                            |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #4).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at
/data1/lesliec/sarthak/mod3-Sarthak-Ti/minitorch/fast_ops.py (331) is hoisted
out of the parallel loop labelled #4 (it will be performed before the loop is
executed and reused inside the loop):
   Allocation:: out_index = np.empty(MAX_DIMS, dtype=np.int32)  # Index buffer
for output
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY

================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply,
/data1/lesliec/sarthak/mod3-Sarthak-Ti/minitorch/fast_ops.py (354)
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /data1/lesliec/sarthak/mod3-Sarthak-Ti/minitorch/fast_ops.py (354)
-------------------------------------------------------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                                                               |
    out: Storage,                                                                                                                          |
    out_shape: Shape,                                                                                                                      |
    out_strides: Strides,                                                                                                                  |
    a_storage: Storage,                                                                                                                    |
    a_shape: Shape,                                                                                                                        |
    a_strides: Strides,                                                                                                                    |
    b_storage: Storage,                                                                                                                    |
    b_shape: Shape,                                                                                                                        |
    b_strides: Strides,                                                                                                                    |
) -> None:                                                                                                                                 |
    """NUMBA tensor matrix multiply function.                                                                                              |
                                                                                                                                           |
    Should work for any tensor shapes that broadcast as long as                                                                            |
                                                                                                                                           |
    ```                                                                                                                                    |
    assert a_shape[-1] == b_shape[-2]                                                                                                      |
    ```                                                                                                                                    |
                                                                                                                                           |
    Optimizations:                                                                                                                         |
                                                                                                                                           |
    * Outer loop in parallel                                                                                                               |
    * No index buffers or function calls                                                                                                   |
    * Inner loop should have no global writes, 1 multiply.                                                                                 |
                                                                                                                                           |
                                                                                                                                           |
    Args:                                                                                                                                  |
    ----                                                                                                                                   |
        out (Storage): storage for `out` tensor                                                                                            |
        out_shape (Shape): shape for `out` tensor                                                                                          |
        out_strides (Strides): strides for `out` tensor                                                                                    |
        a_storage (Storage): storage for `a` tensor                                                                                        |
        a_shape (Shape): shape for `a` tensor                                                                                              |
        a_strides (Strides): strides for `a` tensor                                                                                        |
        b_storage (Storage): storage for `b` tensor                                                                                        |
        b_shape (Shape): shape for `b` tensor                                                                                              |
        b_strides (Strides): strides for `b` tensor                                                                                        |
                                                                                                                                           |
    Returns:                                                                                                                               |
    -------                                                                                                                                |
        None : Fills in `out`                                                                                                              |
                                                                                                                                           |
    """                                                                                                                                    |
    # these simply let us know the strides for the batch, i, j, k                                                                          |
    # the i is the row, j is the column, k is the inner dimension that is shared between a and b                                           |
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                                                                 |
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                                                                 |
    a_i_stride = a_strides[1] if a_shape[1] > 1 else 0                                                                                     |
    b_j_stride = b_strides[2] if b_shape[2] > 1 else 0                                                                                     |
    a_k_stride = a_strides[2] if a_shape[2] > 1 else 0                                                                                     |
    b_k_stride = b_strides[1] if b_shape[1] > 1 else 0                                                                                     |
    # batch_max = max(a_shape[0], b_shape[0])                                                                                              |
                                                                                                                                           |
    # we have to do loops like this because we can't have index buffers                                                                    |
    # also parallelizing over the batch dimension doesn't speed up a large matrix multiply, only when have many batches                    |
    # the basic idea here is we take a batch and assign it to a core. Then each core simply computes the matrix multiply for that batch    |
    for b in prange(out_shape[0]):---------------------------------------------------------------------------------------------------------| #5
        # now we iterate over the other dimensions, pretend we have 2 2x2 matrices                                                         |
        for i in range(out_shape[1]):  # basic matrix multiply loop                                                                        |
            for j in range(out_shape[2]):                                                                                                  |
                tmp = 0.0                                                                                                                  |
                # we also calculate the indices so we don't have to do multiplies in the for loop which only allows 1 multiply             |
                a_index = b * a_batch_stride + i * a_i_stride                                                                              |
                b_index = b * b_batch_stride + j * b_j_stride                                                                              |
                for k in range(a_shape[-1]):                                                                                               |
                    tmp += a_storage[a_index] * b_storage[b_index]                                                                         |
                    a_index += a_k_stride  # add the indices so we don't have to multiply, basically moves by the stride dimensions        |
                    b_index += b_k_stride                                                                                                  |
                                                                                                                                           |
                out_index = (                                                                                                              |
                    b * out_strides[0] + i * out_strides[1] + j * out_strides[2]                                                           |
                )  # now find the out index and write it out                                                                               |
                out[out_index] = tmp                                                                                                       |
                # now loop and find the next element. compute the full product and move on!!                                               |
                                                                                                                                           |
    return None                                                                                                                            |
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #5).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------

---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```


# Matmul speed plot
I used code from Hashim Hayat and Justin Chiu to run this.

### fast vs slow
First I compared simple ops to fast, but the scale of speeds is so far apart it's hard to see the fast compared to the slow, and it would take hours for 1024, so I only compared up to 256 size matrices. Note that in terms of parallelization, the only part of the Fast implementation that is parallel is along the batch dimension, and we only did it with a batch size of 2. But we still see noticeable speedups because we use numba loops which are much faster than python for loops. On top of that things are compiled, so you will still see substantial improvements even with a batch size of 1. But if we have larger batch sizes it would be even faster. Note that in this case size means one dimension of the matrix, so total size is 64x64 for the first matrix, 256x256 for the third one etc.
```console
Timing summary
Size: 64
    fast: 0.00231
    slow: 0.57189
Size: 128
    fast: 0.00978
    slow: 4.46177
Size: 256
    fast: 0.04969
    slow: 35.43919
```

We can also visualize it with a plot

![timing_simple](https://github.com/user-attachments/assets/b1793027-33b3-4f34-9a5c-32261c3adc9c)

### gpu vs fast
I also then compared Fastops to gpu, and you see a sizable speedup, especially as matrices get larger. This might be partially due to the way we only parallelize the fast backend over batches of which the size is 2. However, in general we would expect the GPU to get faster as we get larger, but we are both underutilizing the GPU and not fully utilizing the CPU, so we could optimize both still and it's unclear what the speedup would be, but this relatively naive implementation shows the differences that can be achieved. Also the reason gpu is slower for smaller tensors is because we have to transfer data to the GPU. But as they get larger you see the speedup you get from gpu
```console
Timing summary
Size: 64
    fast: 0.00214
    gpu: 0.00385
Size: 128
    fast: 0.00921
    gpu: 0.00903
Size: 256
    fast: 0.04747
    gpu: 0.03024
Size: 512
    fast: 0.49795
    gpu: 0.12134
Size: 1024
    fast: 3.61113
    gpu: 0.50216
```

Here is the plots comparing them.

![timing](https://github.com/user-attachments/assets/668da833-e0c5-40d8-b57b-56dac7b8f367)


The tests were performed on an Nvidia A100 80GB compared to 8 Intel(R) Xeon(R) Platinum 8462Y+ CPU core

# 3.5
All tests were performed on a Nvidia A100 80GB compared to 8 Intel(R) Xeon(R) Platinum 8462Y+ CPU cores

took about .92 s per epoch for the GPU and 0.12 s per epoch for the CPU. The CPU was faster because we had many cores, and we used all 8. The GPU seems slow because we aren't properly utilizing all the cores. I'm also using a fast HPC for this testing, which means the CPU has substantial memory benefits. When I benchmarked this on my local laptop, I saw the GPU was about 1.1 s but the CPU was closer to 2 s. Part of this is we don't have to transfer data to the CPU which is a heavy slowdown as the weights are not on th eGPU and constantly get trasnferred over and numba complains about it.
## simple dataset
### CPU
Dataset used a hidden size of 100 with the default learning rate of 0.05 and 500 epochs
```console
Epoch  0  loss  4.794541044958728 correct 34
Epoch  10  loss  2.480850255360474 correct 45
Epoch  20  loss  1.742710580060275 correct 43
Epoch  30  loss  1.5764155926700125 correct 45
Epoch  40  loss  1.6982034538341928 correct 50
Epoch  50  loss  1.0592022040361284 correct 50
Epoch  60  loss  3.6003492864452165 correct 45
Epoch  70  loss  2.0669913360692127 correct 49
Epoch  80  loss  0.9660770842676192 correct 50
Epoch  90  loss  1.1251238020268923 correct 50
Epoch  100  loss  1.4025544760805053 correct 50
Epoch  110  loss  1.367990733326769 correct 48
Epoch  120  loss  0.6042545281061102 correct 50
Epoch  130  loss  0.6501440330328618 correct 49
Epoch  140  loss  0.9242977629180551 correct 49
Epoch  150  loss  1.0360576183493677 correct 48
Epoch  160  loss  1.5732692060070792 correct 47
Epoch  170  loss  0.4971599746866164 correct 50
Epoch  180  loss  0.5996415659665786 correct 50
Epoch  190  loss  0.5807114027454106 correct 50
Epoch  200  loss  0.5072081017394219 correct 50
Epoch  210  loss  0.0404435542757332 correct 50
Epoch  220  loss  2.007368775001225 correct 50
Epoch  230  loss  0.21752426592096952 correct 50
Epoch  240  loss  0.1666839008442278 correct 50
Epoch  250  loss  0.040553268715926724 correct 50
Epoch  260  loss  0.3359488381246734 correct 50
Epoch  270  loss  0.2135885427207931 correct 49
Epoch  280  loss  0.46963822972418534 correct 50
Epoch  290  loss  0.020145814072678958 correct 50
Epoch  300  loss  0.6137761434619374 correct 50
Epoch  310  loss  0.5907739661950646 correct 50
Epoch  320  loss  0.508992843734162 correct 50
Epoch  330  loss  0.23957547724224867 correct 50
Epoch  340  loss  0.6872884661433727 correct 50
Epoch  350  loss  0.21775349436654426 correct 50
Epoch  360  loss  0.1888499108987003 correct 50
Epoch  370  loss  0.21595621797164877 correct 50
Epoch  380  loss  0.44726174791112666 correct 50
Epoch  390  loss  0.2750351588430572 correct 50
Epoch  400  loss  0.2870273726787419 correct 50
Epoch  410  loss  0.007289137815457765 correct 50
Epoch  420  loss  0.26798360402012095 correct 50
Epoch  430  loss  0.6723653088084874 correct 49
Epoch  440  loss  0.42288692633792335 correct 50
Epoch  450  loss  0.6051554690407579 correct 50
Epoch  460  loss  0.23204058964510785 correct 50
Epoch  470  loss  0.2989254057147682 correct 50
Epoch  480  loss  0.1622866763451555 correct 50
Epoch  490  loss  0.14573362835195455 correct 50
```

### GPU
Dataset used a hidden size of 100 with the default learning rate of 0.05 and 500 epochs
```console
Epoch  0  loss  5.041792294362406 correct 36
Epoch  10  loss  1.4103937941392946 correct 47
Epoch  20  loss  1.8290620784081715 correct 49
Epoch  30  loss  1.9327544182012701 correct 49
Epoch  40  loss  1.1471988449672237 correct 49
Epoch  50  loss  0.3874295244805885 correct 49
Epoch  60  loss  1.0432634647806112 correct 49
Epoch  70  loss  1.1555524465074067 correct 50
Epoch  80  loss  0.08567120184463595 correct 50
Epoch  90  loss  1.376529162182053 correct 50
Epoch  100  loss  0.2324757760062891 correct 50
Epoch  110  loss  1.0503435587015035 correct 50
Epoch  120  loss  0.7307282952178156 correct 50
Epoch  130  loss  0.01671239442627811 correct 50
Epoch  140  loss  0.07918119248091944 correct 50
Epoch  150  loss  0.8506292051642883 correct 50
Epoch  160  loss  0.0348884967551448 correct 49
Epoch  170  loss  0.19614906495549495 correct 50
Epoch  180  loss  0.3555465546015722 correct 50
Epoch  190  loss  0.29721268823671443 correct 50
Epoch  200  loss  0.37669433137449543 correct 50
Epoch  210  loss  0.010494678609042275 correct 50
Epoch  220  loss  0.6204933987190162 correct 50
Epoch  230  loss  0.5361771172780231 correct 50
Epoch  240  loss  0.002237676177901063 correct 50
Epoch  250  loss  0.3360805549473085 correct 50
Epoch  260  loss  0.19891161612254749 correct 50
Epoch  270  loss  0.005823832728470512 correct 50
Epoch  280  loss  0.7209827143668944 correct 50
Epoch  290  loss  0.3365853000882232 correct 50
Epoch  300  loss  0.008718574622888527 correct 50
Epoch  310  loss  0.13118288668581848 correct 50
Epoch  320  loss  0.09079280897901651 correct 50
Epoch  330  loss  0.13378663178728117 correct 50
Epoch  340  loss  0.016098182409798687 correct 50
Epoch  350  loss  0.0018523163325214974 correct 50
Epoch  360  loss  0.14684448482920978 correct 50
Epoch  370  loss  0.267176312938944 correct 50
Epoch  380  loss  0.028208611323019182 correct 50
Epoch  390  loss  0.19026956740851994 correct 50
Epoch  400  loss  0.4401464549203257 correct 50
Epoch  410  loss  0.018081600729691585 correct 50
Epoch  420  loss  0.1354434465080285 correct 50
Epoch  430  loss  0.31209244675334097 correct 50
Epoch  440  loss  0.32017522538075305 correct 50
Epoch  450  loss  0.008536714384211015 correct 50
Epoch  460  loss  0.1508530284372788 correct 50
Epoch  470  loss  0.44190446578588516 correct 50
Epoch  480  loss  0.2560555883824389 correct 50
Epoch  490  loss  0.12461209211837704 correct 50
```


## split dataset
### CPU
Dataset used a hidden size of 100 with the default learning rate of 0.05 and 500 epochs
```console
Epoch  0  loss  11.245732730569863 correct 37
Epoch  10  loss  5.650743420765901 correct 38
Epoch  20  loss  3.7095133058358893 correct 41
Epoch  30  loss  2.9716191594602828 correct 42
Epoch  40  loss  3.9716671341266694 correct 45
Epoch  50  loss  2.4098407720505106 correct 48
Epoch  60  loss  2.7950931705100883 correct 45
Epoch  70  loss  1.790207724636235 correct 49
Epoch  80  loss  2.2237664114392475 correct 48
Epoch  90  loss  3.387671679827898 correct 48
Epoch  100  loss  1.6187483956014699 correct 48
Epoch  110  loss  1.547145933939363 correct 49
Epoch  120  loss  2.6085153492867383 correct 49
Epoch  130  loss  1.9850381471978797 correct 49
Epoch  140  loss  0.5225470768787213 correct 49
Epoch  150  loss  0.5843529208788351 correct 47
Epoch  160  loss  2.45543397075802 correct 47
Epoch  170  loss  0.5869337657631205 correct 49
Epoch  180  loss  0.46229322298655207 correct 49
Epoch  190  loss  1.4872616335384974 correct 49
Epoch  200  loss  0.852138084163072 correct 48
Epoch  210  loss  2.4372069335827256 correct 48
Epoch  220  loss  1.6691728474943113 correct 50
Epoch  230  loss  0.5274782435732537 correct 50
Epoch  240  loss  1.5513394842944737 correct 50
Epoch  250  loss  0.49241036518372366 correct 50
Epoch  260  loss  0.3247151777223071 correct 47
Epoch  270  loss  2.504682601583812 correct 48
Epoch  280  loss  1.339964617853126 correct 47
Epoch  290  loss  0.2119476211147711 correct 49
Epoch  300  loss  0.46599931468764094 correct 49
Epoch  310  loss  0.8694279909834943 correct 50
Epoch  320  loss  0.5075287156128001 correct 49
Epoch  330  loss  1.3308461339921778 correct 49
Epoch  340  loss  0.44961529264424427 correct 49
Epoch  350  loss  0.20399360398892297 correct 49
Epoch  360  loss  1.2601044891387243 correct 49
Epoch  370  loss  1.9724926794144277 correct 48
Epoch  380  loss  1.971566365937519 correct 49
Epoch  390  loss  0.27925341616430743 correct 49
Epoch  400  loss  0.8987279110163862 correct 49
Epoch  410  loss  0.14991060973978432 correct 50
Epoch  420  loss  0.802025389877427 correct 49
Epoch  430  loss  0.16013174512476253 correct 49
Epoch  440  loss  0.3715704494547434 correct 50
Epoch  450  loss  0.09996909186204768 correct 49
Epoch  460  loss  0.10490462169004638 correct 49
Epoch  470  loss  0.03741577304162221 correct 49
Epoch  480  loss  1.310656199134171 correct 50
Epoch  490  loss  0.3025891145409685 correct 50
```

### GPU
Dataset used a hidden size of 100 with the default learning rate of 0.05 and 500 epochs
```console
Epoch  0  loss  8.04287540911433 correct 30
Epoch  10  loss  5.326711668206477 correct 32
Epoch  20  loss  3.8707265487810045 correct 42
Epoch  30  loss  8.723164448263443 correct 44
Epoch  40  loss  3.0643525183174876 correct 50
Epoch  50  loss  2.6775621850504168 correct 49
Epoch  60  loss  2.636092127754758 correct 47
Epoch  70  loss  1.347036364632607 correct 50
Epoch  80  loss  1.5926311162972366 correct 49
Epoch  90  loss  2.6810905293839804 correct 47
Epoch  100  loss  2.2970243196626488 correct 48
Epoch  110  loss  1.438576253725656 correct 50
Epoch  120  loss  1.28003260529785 correct 50
Epoch  130  loss  0.5876232879620573 correct 50
Epoch  140  loss  0.5670210795521853 correct 50
Epoch  150  loss  0.531079885915537 correct 50
Epoch  160  loss  0.4987883199032385 correct 50
Epoch  170  loss  0.9789175026751139 correct 50
Epoch  180  loss  0.6067380656987728 correct 50
Epoch  190  loss  0.5816616915682417 correct 50
Epoch  200  loss  0.050364875931734596 correct 50
Epoch  210  loss  0.25621722246047274 correct 50
Epoch  220  loss  0.5150573220048068 correct 50
Epoch  230  loss  0.11891392274300143 correct 50
Epoch  240  loss  0.4611466718291381 correct 50
Epoch  250  loss  0.31735751783498567 correct 50
Epoch  260  loss  0.11729266631962408 correct 50
Epoch  270  loss  0.4906452579346145 correct 50
Epoch  280  loss  0.4098205160061711 correct 50
Epoch  290  loss  0.3719328912449623 correct 50
Epoch  300  loss  0.3484447519942949 correct 50
Epoch  310  loss  0.1513766862525347 correct 50
Epoch  320  loss  0.3882383145656866 correct 50
Epoch  330  loss  0.12925647009970628 correct 50
Epoch  340  loss  0.1151768409380462 correct 50
Epoch  350  loss  0.1860132011882316 correct 50
Epoch  360  loss  0.2284872821597223 correct 50
Epoch  370  loss  0.0351713842213731 correct 50
Epoch  380  loss  0.09190131196061681 correct 50
Epoch  390  loss  0.18285236904177793 correct 50
Epoch  400  loss  0.21296265909380202 correct 50
Epoch  410  loss  0.12391413220722475 correct 50
Epoch  420  loss  0.14242647091309749 correct 50
Epoch  430  loss  0.11029272375549863 correct 50
Epoch  440  loss  0.0992450456991964 correct 50
Epoch  450  loss  0.1675697169081874 correct 50
Epoch  460  loss  0.11981929759520846 correct 50
Epoch  470  loss  0.07251280966220398 correct 50
Epoch  480  loss  0.23556617542720493 correct 50
Epoch  490  loss  0.14117786864790138 correct 50
```

## XOR dataset

### CPU
Dataset used a hidden size of 100 with the default learning rate of 0.05 and 500 epochs
```console
Epoch  0  loss  5.310291172032426 correct 27
Epoch  10  loss  5.355725625288696 correct 44
Epoch  20  loss  3.1732513425445044 correct 46
Epoch  30  loss  3.3714911304242166 correct 46
Epoch  40  loss  3.2639771319844932 correct 44
Epoch  50  loss  2.725334840182191 correct 46
Epoch  60  loss  1.8025567348283036 correct 46
Epoch  70  loss  2.3034696117491915 correct 47
Epoch  80  loss  1.7643910015598134 correct 46
Epoch  90  loss  4.3071198922646365 correct 46
Epoch  100  loss  3.813697534149517 correct 46
Epoch  110  loss  2.6390648259967513 correct 47
Epoch  120  loss  1.9954072623890649 correct 47
Epoch  130  loss  2.3929904496017342 correct 48
Epoch  140  loss  3.762469165300859 correct 48
Epoch  150  loss  3.8450810217755134 correct 46
Epoch  160  loss  1.1986779347852632 correct 47
Epoch  170  loss  2.4462551154116072 correct 47
Epoch  180  loss  1.1237253887821161 correct 47
Epoch  190  loss  1.4553170050744895 correct 47
Epoch  200  loss  3.210013102836976 correct 47
Epoch  210  loss  0.6246086010410321 correct 47
Epoch  220  loss  1.2328808013090506 correct 47
Epoch  230  loss  1.1273041968000261 correct 47
Epoch  240  loss  1.6467133796297038 correct 48
Epoch  250  loss  0.17163316340569015 correct 46
Epoch  260  loss  1.2324985062167713 correct 47
Epoch  270  loss  3.629967620512794 correct 47
Epoch  280  loss  0.18873849033677317 correct 47
Epoch  290  loss  1.354784621590528 correct 49
Epoch  300  loss  0.33641660615791985 correct 47
Epoch  310  loss  2.172653471725714 correct 46
Epoch  320  loss  1.6195890516669629 correct 47
Epoch  330  loss  0.9590337896712526 correct 47
Epoch  340  loss  1.755821127096972 correct 48
Epoch  350  loss  0.2575968146183404 correct 49
Epoch  360  loss  0.3760784798704119 correct 47
Epoch  370  loss  0.815449220877628 correct 48
Epoch  380  loss  2.1764155748114673 correct 47
Epoch  390  loss  0.8671271822484872 correct 47
Epoch  400  loss  0.16601158984448683 correct 47
Epoch  410  loss  0.7492618094076422 correct 47
Epoch  420  loss  0.5690993379086704 correct 47
Epoch  430  loss  1.4331353462694203 correct 49
Epoch  440  loss  1.8290663710990063 correct 47
Epoch  450  loss  1.1516356809511799 correct 49
Epoch  460  loss  0.7542170913023896 correct 49
Epoch  470  loss  0.7619404270764527 correct 50
Epoch  480  loss  0.6720228764384392 correct 49
Epoch  490  loss  0.850379710740381 correct 50
```

### GPU
Dataset used a hidden size of 100 with the default learning rate of 0.05 and 500 epochs
```console
Epoch  0  loss  5.033688573780841 correct 29
Epoch  10  loss  4.665444345282475 correct 39
Epoch  20  loss  5.04563733382302 correct 45
Epoch  30  loss  3.1401601735381925 correct 47
Epoch  40  loss  4.540658873232579 correct 46
Epoch  50  loss  3.366803169747932 correct 46
Epoch  60  loss  3.381184552570685 correct 47
Epoch  70  loss  1.7935301196467373 correct 43
Epoch  80  loss  2.6401095043816687 correct 47
Epoch  90  loss  2.2637941014245673 correct 46
Epoch  100  loss  1.7048745203396178 correct 49
Epoch  110  loss  2.133319010115957 correct 48
Epoch  120  loss  2.0529981512680133 correct 46
Epoch  130  loss  1.4116665107233133 correct 49
Epoch  140  loss  1.141484443723891 correct 47
Epoch  150  loss  1.1165282874653784 correct 46
Epoch  160  loss  1.2353449281314033 correct 49
Epoch  170  loss  0.7080609104670134 correct 47
Epoch  180  loss  1.5688051099633153 correct 47
Epoch  190  loss  1.0936021979602804 correct 49
Epoch  200  loss  2.491871983694268 correct 48
Epoch  210  loss  1.1951708841325923 correct 47
Epoch  220  loss  0.6835695501646062 correct 50
Epoch  230  loss  1.360095922561091 correct 49
Epoch  240  loss  2.009417754508564 correct 49
Epoch  250  loss  0.9573301001640435 correct 50
Epoch  260  loss  0.6184394729535215 correct 50
Epoch  270  loss  0.7078922094716349 correct 49
Epoch  280  loss  0.4339353326465828 correct 49
Epoch  290  loss  0.7597206935214585 correct 48
Epoch  300  loss  1.2598020345478513 correct 50
Epoch  310  loss  0.4257006676631018 correct 50
Epoch  320  loss  0.8882916818260187 correct 50
Epoch  330  loss  0.48421313939594646 correct 49
Epoch  340  loss  1.6577482212996721 correct 48
Epoch  350  loss  0.14291843951645752 correct 47
Epoch  360  loss  0.3274322030580512 correct 50
Epoch  370  loss  1.1895447649161188 correct 49
Epoch  380  loss  0.2535884799101773 correct 50
Epoch  390  loss  0.6146076441684162 correct 50
Epoch  400  loss  0.47239573819706426 correct 50
Epoch  410  loss  0.1524288887427605 correct 49
Epoch  420  loss  0.5106787634272131 correct 50
Epoch  430  loss  1.077143491406866 correct 49
Epoch  440  loss  0.09599011664834277 correct 50
Epoch  450  loss  0.23511692649197358 correct 50
Epoch  460  loss  1.7100054954194515 correct 49
Epoch  470  loss  0.556319151138438 correct 50
Epoch  480  loss  0.8069134924796677 correct 48
Epoch  490  loss  1.1276323386462963 correct 50
```


## Benchmarking large model
I used a hidden layer size of 1000, I modified the trainer to also print the average time for the last 10 epochs, and only trained for 200 epochs on the simple dataset

For this test I used an A100 80 GB GPU and 8 Intel(R) Xeon(R) Platinum 8462Y+ CPU cores on the IRIS cluster at MSK. Part of the issue with the speed is that we don't use most of the GPU, numba keeps giving performance warnings, but use of that is outside of my control. Also we don't transfer data to the GPU which means we constantly transfer between GPU and CPU which is very slow! Finally, the IRIS cluster has reallyy fast CPU memory access but it's much slower for the GPU, so these constant read and writes will be significant slowdowns, but even then, it's very fast. IF it was on the GPU we'd see heavy speed increases as we see in the large matmul operations which are slow because there's a lot of operations, here the major slowdown is the transferring between cpu and gpu memory.

We also see that the GPU doesn't slow down much compared to the CPU when we go from hidden size 100 to 1000. The slowdown for the CPU is like a factor of 100, but for the GPU is barely 2x, becaukse the main bottleneck was transferring data, not the calculations. We can see this with the benchmarking on the matmul where massive matrices were much faster for th eGPU because once the data was transferred over, the one operation was very quick!
### CPU
Averaged about 1.17 s per epoch. Each print tells you the average time of the last 10 epochs
```console
Epoch 0 | Loss: 131.9095 | Correct: 32 | Average time per epoch: 0.0000s
Epoch 10 | Loss: 1.2593 | Correct: 38 | Average time per epoch: 1.1683s
Epoch 20 | Loss: 1.6557 | Correct: 42 | Average time per epoch: 1.1647s
Epoch 30 | Loss: 0.1249 | Correct: 50 | Average time per epoch: 1.1680s
Epoch 40 | Loss: 0.0107 | Correct: 50 | Average time per epoch: 1.1709s
Epoch 50 | Loss: 0.4487 | Correct: 49 | Average time per epoch: 1.1735s
Epoch 60 | Loss: 0.4647 | Correct: 50 | Average time per epoch: 1.1685s
Epoch 70 | Loss: 0.4176 | Correct: 50 | Average time per epoch: 1.1725s
Epoch 80 | Loss: 0.3629 | Correct: 50 | Average time per epoch: 1.1709s
Epoch 90 | Loss: 1.0943 | Correct: 48 | Average time per epoch: 1.1698s
Epoch 100 | Loss: 0.4081 | Correct: 49 | Average time per epoch: 1.1696s
Epoch 110 | Loss: 0.0008 | Correct: 50 | Average time per epoch: 1.1697s
Epoch 120 | Loss: 0.0029 | Correct: 50 | Average time per epoch: 1.1689s
Epoch 130 | Loss: 0.2554 | Correct: 50 | Average time per epoch: 1.1707s
Epoch 140 | Loss: 0.1657 | Correct: 50 | Average time per epoch: 1.1732s
Epoch 150 | Loss: 0.4295 | Correct: 50 | Average time per epoch: 1.1715s
Epoch 160 | Loss: 0.3672 | Correct: 50 | Average time per epoch: 1.1728s
Epoch 170 | Loss: 0.0002 | Correct: 50 | Average time per epoch: 1.1686s
Epoch 180 | Loss: 0.0784 | Correct: 50 | Average time per epoch: 1.1745s
Epoch 190 | Loss: 0.0001 | Correct: 50 | Average time per epoch: 1.1717s
```

### GPU
Averaged 1.77s per epoch for this really large dataset. This is actually quite fast especially since we're not fully utilizing the GPU and the major slowdown is transfer of data to the GPU
```console
Epoch 0 | Loss: 45.0684 | Correct: 27 | Time since last call: NaN
Epoch 10 | Loss: 3.1021 | Correct: 41 | Average time per epoch: 1.7718s
Epoch 20 | Loss: 1.0329 | Correct: 46 | Average time per epoch: 1.7750s
Epoch 30 | Loss: 0.0075 | Correct: 50 | Average time per epoch: 1.7750s
Epoch 40 | Loss: 0.0521 | Correct: 50 | Average time per epoch: 1.7717s
Epoch 50 | Loss: 0.0244 | Correct: 50 | Average time per epoch: 1.7733s
Epoch 60 | Loss: 0.1336 | Correct: 50 | Average time per epoch: 1.7627s
Epoch 70 | Loss: 0.1092 | Correct: 50 | Average time per epoch: 1.7719s
Epoch 80 | Loss: 0.0065 | Correct: 50 | Average time per epoch: 1.7679s
Epoch 90 | Loss: 0.0296 | Correct: 50 | Average time per epoch: 1.7740s
Epoch 100 | Loss: 0.0075 | Correct: 50 | Average time per epoch: 1.7723s
Epoch 110 | Loss: 0.0328 | Correct: 50 | Average time per epoch: 1.7642s
Epoch 120 | Loss: 0.0014 | Correct: 50 | Average time per epoch: 1.7746s
Epoch 130 | Loss: 0.0067 | Correct: 50 | Average time per epoch: 1.7726s
Epoch 140 | Loss: 0.0479 | Correct: 50 | Average time per epoch: 1.7812s
Epoch 150 | Loss: 0.0509 | Correct: 50 | Average time per epoch: 1.7774s
Epoch 160 | Loss: 0.0462 | Correct: 50 | Average time per epoch: 1.7707s
Epoch 170 | Loss: 0.0452 | Correct: 50 | Average time per epoch: 1.7645s
Epoch 180 | Loss: 0.0534 | Correct: 50 | Average time per epoch: 1.7725s
Epoch 190 | Loss: 0.0219 | Correct: 50 | Average time per epoch: 1.7704s
```

### simple
We also compared it to the simple non parallel non compiled old tensor backend on a hidden layer size of 1000. We only did 10 epochs as it is extremely ridiculously slow!! Even the 0 epoch took almost 5 minutes, the backward in particular was very very slow!! I don't want to submit a super long job to do this basic unoptimized calculation. But we can see it's a SIGNIFICANT slowdown compared to both numba and gpu, and that's the whole point of DL being heavily parallelizable and significantly sped up by parallelizing!! Also this required me create a very slow matrix multiply function in tensor_ops, but it was just an easier version to write than the fast tensor version for numba. But it takes almost 5 minutes PER EPOCH!!! This is ridiculously slow and shows how much more efficient we have written our code!
```console
Epoch 0 | Loss: 0.2237 | Correct: 44 | Average time per epoch: 0.0000s
Epoch 10 | Loss: 0.0185 | Correct: 50 | Average time per epoch: 289.4307s
```



# base readme
You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py
