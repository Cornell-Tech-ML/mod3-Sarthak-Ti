# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


# Matmul speed plot
I used code from Hashim Hayat and Justin Chiu to run this.

First I compared simple ops to fast, but the scale of speeds is so far apart it's hard to see the fast compared to the slow, and it would take hours for 1024, so I only compared up to 256 size matrices
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



I also then compared Fastops to gpu, and you see a sizable speedup, especially as matrices get larger
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


The tests were performed on an Nvidia A100 80GB compared to 8 CPU core

# 3.5
All tests were performed on a Nvidia A100 80GB compared to 8 CPU cores

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
I used a hidden layer size of 1000, I modified the trainer to also print the time for the 10 epochs, and only trained for 200 epochs on xor dataset

For this test I used an A100 80 GB GPU and 8 CPU cores on the IRIS cluster at MSK. PArt of the issue with the speed is that we don't use most of the GPU, numba keeps giving performance warnings, but use of that is outside of my control. Also we don't transfer data to the GPU which means we constantly transfer between GPU and CPU which is very slow! Finally, the IRIS cluster has reallyy fast CPU memory access but it's much slower for the GPU, so these constant read and writes will be significant slowdowns, but even then, it's very fast. IF it was on the GPU we'd see heavy speed increases as we see in the large matmul operations which are slow because

We also see that the GPU doesn't slow down much compared to the CPU when we go from hidden size 100 to 1000. The slowdown for the CPU is like a factor of 100, but for the GPU is barely 2x, becaukse the main bottleneck was transferring data, not the calculations. We can see this with the benchmarking on the matmul where massive matrices were much faster for th eGPU because once the data was transferred over, the one operation was very quick!
### CPU
Averaged less than 2s per epoch. EAch print tells you the time between calls and since each call is 10 epochs, the timne between epochs being less than 20 implies it's faster than 10 s.
```console
Epoch 0 | Loss: 575.6463 | Correct: 26 | Time since last call: NaN
Epoch 10 | Loss: 76.6751 | Correct: 43 | Time since last call: 19.76s
Epoch 20 | Loss: 0.3261 | Correct: 47 | Time since last call: 19.72s
Epoch 30 | Loss: 0.0565 | Correct: 48 | Time since last call: 19.71s
Epoch 40 | Loss: 2.2168 | Correct: 45 | Time since last call: 19.75s
Epoch 50 | Loss: 0.1740 | Correct: 50 | Time since last call: 19.79s
Epoch 60 | Loss: 0.0411 | Correct: 50 | Time since last call: 19.77s
Epoch 70 | Loss: 0.6611 | Correct: 50 | Time since last call: 19.80s
Epoch 80 | Loss: 0.8915 | Correct: 49 | Time since last call: 19.73s
Epoch 90 | Loss: 0.1637 | Correct: 49 | Time since last call: 19.76s
Epoch 100 | Loss: 0.0540 | Correct: 46 | Time since last call: 19.77s
Epoch 110 | Loss: 0.5228 | Correct: 50 | Time since last call: 19.75s
Epoch 120 | Loss: 0.0070 | Correct: 50 | Time since last call: 19.79s
Epoch 130 | Loss: 0.9954 | Correct: 47 | Time since last call: 19.76s
Epoch 140 | Loss: 0.0086 | Correct: 50 | Time since last call: 19.79s
Epoch 150 | Loss: 0.4423 | Correct: 50 | Time since last call: 19.81s
Epoch 160 | Loss: 0.7900 | Correct: 49 | Time since last call: 19.79s
Epoch 170 | Loss: 0.0093 | Correct: 50 | Time since last call: 19.78s
Epoch 180 | Loss: 0.0102 | Correct: 50 | Time since last call: 19.82s
Epoch 190 | Loss: 0.3353 | Correct: 50 | Time since last call: 19.79s
```

### GPU
Averaged 2.8s per epoch for this really large dataset. This is actually quite fast
Epoch 0 | Loss: 45.0684 | Correct: 27 | Time since last call: NaN
Epoch 10 | Loss: 3.1021 | Correct: 41 | Time since last call: 28.12s
Epoch 20 | Loss: 1.0329 | Correct: 46 | Time since last call: 27.97s
Epoch 30 | Loss: 0.0014 | Correct: 50 | Time since last call: 27.95s
Epoch 40 | Loss: 0.0000 | Correct: 50 | Time since last call: 27.94s
Epoch 50 | Loss: 0.0011 | Correct: 50 | Time since last call: 27.93s
Epoch 60 | Loss: 0.0022 | Correct: 50 | Time since last call: 27.87s
Epoch 70 | Loss: 0.0001 | Correct: 50 | Time since last call: 27.94s
Epoch 80 | Loss: 0.0000 | Correct: 50 | Time since last call: 27.90s
Epoch 90 | Loss: 0.0000 | Correct: 50 | Time since last call: 27.94s
Epoch 100 | Loss: 0.0019 | Correct: 50 | Time since last call: 27.92s
Epoch 110 | Loss: 0.0038 | Correct: 50 | Time since last call: 27.94s
Epoch 120 | Loss: 0.0001 | Correct: 50 | Time since last call: 27.87s
Epoch 130 | Loss: 0.0008 | Correct: 50 | Time since last call: 27.89s
Epoch 140 | Loss: 0.0001 | Correct: 50 | Time since last call: 27.91s
Epoch 150 | Loss: 0.0000 | Correct: 50 | Time since last call: 27.91s
Epoch 160 | Loss: 0.0007 | Correct: 50 | Time since last call: 27.90s
Epoch 170 | Loss: 0.0000 | Correct: 50 | Time since last call: 27.88s
Epoch 180 | Loss: 0.0015 | Correct: 50 | Time since last call: 27.92s
Epoch 190 | Loss: 0.0000 | Correct: 50 | Time since last call: 27.91s


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