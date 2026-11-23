# PyTorch image classification examples

These two independent programs train a ResNet-18 image classifier:

- `train_imagenet.py` loads an extracted ImageNet-style dataset with `torchvision.datasets.ImageFolder` and trains from random initialization.
- `train_cifar10.py` downloads CIFAR-10 through `torchvision.datasets.CIFAR10` and trains from random initialization.

## 1. Install PyTorch and torchvision

Use the official [PyTorch installation selector](https://pytorch.org/get-started/locally/) to choose the command for your operating system and CPU/GPU. Then install the other requirement:

```powershell
python -m pip install pillow
```

These scripts use the current PyTorch 2.x `torch.amp` API. Run commands below from this directory, or pass paths relative to your current directory.

## 2. Train on ImageNet

ImageNet is not downloaded by this script. Obtain it through the official ImageNet access process, then arrange it in class folders:

```text
imagenet/
  train/
    n01440764/
      image1.JPEG
    n01443537/
      image2.JPEG
  val/
    n01440764/
      image3.JPEG
    n01443537/
      image4.JPEG
```

The actual dataset has many more class folders. The training and validation folders must use the same class names. The script discovers the classes and sets the model's final layer to match them. In particular, the validation images must already be sorted into class subfolders for `ImageFolder` to read their labels.

Start a standard 90-epoch run:

```powershell
python train_imagenet.py --data D:/datasets/imagenet --output checkpoints/imagenet --amp
```

For a small smoke-sized run, reduce the batch size and number of epochs:

```powershell
python train_imagenet.py --data D:/datasets/imagenet --epochs 1 --batch-size 16 --workers 0
```

## 3. Train on CIFAR-10

The CIFAR-10 archives download automatically into `data/` the first time you run the script:

```powershell
python train_cifar10.py --epochs 50 --batch-size 128 --amp
```

The default device selection chooses CUDA, then Apple MPS, then CPU. Pass `--device cpu`, `--device cuda`, or `--device mps` to select one explicitly. `--amp` enables automatic mixed precision only on CUDA.

## Outputs and training choices

Each program writes `last.pt` after every epoch and `best.pt` when validation accuracy improves. Checkpoints contain model and optimizer state, scheduler state, class names, and epoch number. For an inference-only model, construct the same model architecture and load `checkpoint["model_state_dict"]`.

Both models use `weights=None`, so training starts from random weights. ImageNet uses the standard 224-pixel ResNet-18 input pipeline. CIFAR-10 uses a 32-pixel ResNet-18 stem plus random crop and horizontal-flip augmentation. The CIFAR-10 test set is evaluated each epoch for a simple example; for rigorous final reporting, keep a separate validation set and evaluate the official test set only after tuning is finished.

## 4. Tiny GPT using NumPy only

[`gpt_numpy_toy.py`](gpt_numpy_toy.py) is an original, trainable character-level model inspired by the causal attention, next-token objective, and autoregressive sampling in [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/). I let this version run longer than 60 lines so each operation is easier to follow. It has one causal attention head, a residual connection, manual NumPy gradients, and simple SGD. To keep the core visible, it leaves out multi-head attention, LayerNorm, and the feed-forward sublayer. It trains on a tiny built-in sample corpus; it does not load GPT-2's pretrained weights or use GPT-2's tokenizer.

The main steps are:

1. Convert each character into an integer token.
2. Add token and position embeddings.
3. Compute attention with `softmax(Q @ K.T / sqrt(d))`, masking future positions.
4. Use the attention result to predict the next character at every position.
5. Compare predictions with the text shifted by one character, then update the weights.
6. Generate by sampling a character and appending it to the context repeatedly.

Install NumPy if needed, then run:

```powershell
python -m pip install numpy
python gpt_numpy_toy.py
```
