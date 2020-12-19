# A-comprehensive-evaluation-of-the-sentiment-analysis-on-the-cloud

Ashwin Nalwade, Mingxi Chen. 

# Comparison across different platforms

We test out the different approaches across 3 different cloud platforms, and we analyze the
training times [average over epochs], accuracies, and memory utilizations [peak value] by using
profilers while running on the GPU.

![GPU](https://github.com/ashwinpn/A-comprehensive-evaluation-of-the-sentiment-analysis-on-the-cloud/blob/main/resources/gpu_comparison.png)

# GPU Details [Colab Pro]
```bash
Fri Nov 27 20:47:34 2020
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 455.38 Driver Version: 418.67 CUDA Version: 10.1 |
|-------------------------------+----------------------+----------------------+
| GPU Name Persistence-M| Bus-Id Disp.A | Volatile Uncorr. ECC |
| Fan Temp Perf Pwr:Usage/Cap| Memory-Usage | GPU-Util Compute M. |
| | | MIG M. |
|===============================+======================+======================|
| 0 Tesla V100-SXM2... Off | 00000000:00:04.0 Off | 0 |
| N/A 36C P0 38W / 300W | 3235MiB / 16130MiB | 0% Default |
| | | ERR! |
+-------------------------------+----------------------+----------------------+
```
