# A-comprehensive-evaluation-of-the-sentiment-analysis-on-the-cloud

Ashwin Nalwade, Mingxi Chen. 

## Comparison across different platforms

We test out the different approaches across 3 different cloud platforms, and we analyze the
training times [average over epochs], accuracies, and memory utilizations [peak value] by using
profilers while running on the GPU.

![GPU](https://github.com/ashwinpn/A-comprehensive-evaluation-of-the-sentiment-analysis-on-the-cloud/blob/main/resources/gpu_comparison.png)

## GPU Details [Colab Pro]
```bash
Fri Nov 27 20:47:34 2020
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 455.45.01    Driver Version: 418.67       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  Off  | 00000000:00:04.0 Off |                    0 |
| N/A   41C    P0    26W / 250W |      0MiB / 16280MiB |      0%      Default |
|                               |                      |                 ERR! |
+-------------------------------+----------------------+----------------------+
```

## Knowledge Distillation

![Kd](https://github.com/ashwinpn/A-comprehensive-evaluation-of-the-sentiment-analysis-on-the-cloud/blob/main/resources/kd.png)

## Deploying the web application using Kubernetes
Deployed on the Google Kubernetes Engine.
We deploy it as a web application with
a ```LoadBalancer``` class and ```3 replicaSets``` set the configuration in the deployment .yaml file],
which ensures both scalability and reliability. That is, with a ```Loadbalancer``` the traffic would be
evenly distributed across the 3 pods [scalability], and even if one of the pods is not available,
users can access a successfully running web application due to the other functional pods
[reliability]. We also expose the application so that it can be accessed externally.



```bash
kubectl create -f dply.yaml

kubectl get pods

kubectl get deployments

kubectl expose deployment sentiment-inference-service --type=LoadBalancer --port 80 --target-port 8080
```

![demo](https://github.com/ashwinpn/A-comprehensive-evaluation-of-the-sentiment-analysis-on-the-cloud/blob/main/resources/caml_demo.gif)
