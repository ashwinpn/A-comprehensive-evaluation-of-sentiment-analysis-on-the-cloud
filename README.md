# A-comprehensive-evaluation-of-sentiment-analysis-on-the-cloud

Ashwin Nalwade, Mingxi Chen. 

## Tech Stack
Application and Data: Python, Flask, Gunicorn, CSS, spaCy, PyTorch, Pandas, HuggingFace.

Cloud : Google Cloud Platform (GCP), IBM Cloud, Gradient Cloud by Paperspace.

Containers : Docker [Docker Hub], Kubernetes, Google Kubernetes Engine.

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

## Neural Network Architecture Decisions
### Recurrent Neural Networks
- We make use of packed padded sequences here, which would enable our Recurrent
Neural Network to process only that part of the sentence which is not padded [and for
the padded parts the output will be a zero tensor].

- We make use of RNN architecture known as Long Short Term Memory, acronymed
LSTM. LSTM’s are useful here because they do not suffer from the vanishing gradient
problem, which your usual RNN’s do. LSTM’s leverage an auxiliary state called a cell
which is considered analogous to the memory of the LSTM. They also make use of
numerous gates which manage the flow of information from and to the cell.

- <ins>Regularization</ins>. Having a large number of parameters in the model implies that we
would have a larger probability of the occurrence of overfitting [That is, train too closely
on the training data, leading to a high training accuracy and low train error BUT lower
test, validation accuracies and high test, validation errors]. Thus, regularization is crucial
to prevent this from occurring. There are various regularization methods like lasso / ridge
regression, (l1+l2) regression, but we use dropout. Dropout operates by randomly
deleting neurons within a layer in a forward pass.

### Transformers
- We use the BERT (Bidirectional Encoder Representations from Transformers)
Transformer model, first developed by Devlin et. al

- We make use of the transformers as our embedding layers, and then train the other part
of the model - it will learn from the representations created by the transformer. We
leverage a bidirectional gated recurrent unit (GRU) for this.

- The transformers, utilized as embeddings, are forwarded to a GRU to generate
sentiment predictions for the test sentence.

## Knowledge Distillation

![Kd](https://github.com/ashwinpn/A-comprehensive-evaluation-of-the-sentiment-analysis-on-the-cloud/blob/main/resources/kd.png)


- Knowledge distillation is a method by which a small model is trained to imitate the
behaviour of a large model, seeking to reproduce similar results as the larger model.
Basically, it compresses the bigger model [the teacher] into a smaller model [student].

- When thinking about real world applications, successfully running smaller models with
similar performance metrics and accuracy values would enable us to ensure that
machine learning / NLP based web applications which require large models to perform
inference can function seamlessly on mobile devices too [Otherwise we might need
costly GPU servers to maintain scalability].


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

## Parameter Tuning
```bash
====================================
Dropout = 0.5:
peak memory: 2357.59 MiB, increment: 20.79 MiB
Epoch: 05 | Epoch Time: 0m 30s
Train Loss: 0.074 | Train Acc: 97.57%
Val. Loss: 0.371 | Val. Acc: 87.51%
Test Loss: 0.341 | Test Acc: 85.46%

Dropout = 0.2:
peak memory: 2448.95 MiB, increment: 0.10 MiB
Epoch: 05 | Epoch Time: 0m 30s
Train Loss: 0.047 | Train Acc: 98.97%
 Val. Loss: 0.320 | Val. Acc: 88.28%
Test Loss: 0.327 | Test Acc: 85.82%

Dropout = 0.1:
peak memory: 2468.17 MiB, increment: 0.00 MiB
Epoch: 05 | Epoch Time: 0m 30s
Train Loss: 0.032 | Train Acc: 99.54%
Val. Loss: 0.317 | Val. Acc: 88.44%
Test Loss: 0.313 | Test Acc: 86.83%
```


## Experiments
- We make use of the DistilBERT library provided by HuggingFace, one of the leading
organisations working on NLP and transformers. Compared to the original BERT, which
has 112,241,409 trainable parameters, DistilBERT has 69,122,049 trainable parameters
[A reduction of <ins>38.41%</ins>]. This is impressive, considering the fact that we were able to
reduce the average training time per epoch by <ins>45.59%</ins> [almost half] at the expense of
just <ins>1.16%</ins> reduction in the test accuracy.

- We built an app for inference [with flask and gunicorn], ran it locally, packaged it
using docker, uploaded the dockerfile to Docker Hub and deployed it with kubernetes - also ensuring scalability and
reliability using LoadBalancer and ReplicaSets. We exposed it externally, so
that it can be accessed from anywhere, and provide it with a smooth UI.

- Compared to other models, a big argument for CNN’s is that they were found to be fast -
not just for training, but even for inference. Theoretically, this makes sense as
convolutions are a central part of computer graphics and implemented on a hardware
level on GPU’s. In quantitative terms, in the time that it takes to train 1 epoch for a
transformer [4m 21s], we could have trained a CNN for ~17 epochs [and reached
convergence]. Also, the saved model [.pth] files for a CNN are 10.0 MiB, compared to
428 MiB for a transformer.

