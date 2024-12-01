# A Quick Ctrl-G Setup Guide
The [Ctrl-G](https://github.com/joshuacnf/Ctrl-G/tree/main) repository has excellent instructions on how to set up your environment to start generating text
with Ctrl-G, but there are some things you need to be aware of. First of all, the default beam size and batch size
is way too large, and will most likely result in memory errors unless you have a very powerful computer.
I reccomend setting the batch size to 1 and adjusting the beam size as needed.
Secondly, some of the packages are Linux exclusive, so they won't work on Windows.

However, after some trial and error, we were able to set up Ctrl-G locally and on Google Colab.

## Google Colab
I've created a notebook on Google Colab that you can clone and edit as you wish. Since Ctrl-G isn't packaged,
I had to get a bit creative when using it on Google Colab. I went through a few different iterations, but this
ended up being the simplest [Link](https://colab.research.google.com/drive/1Gcp16pz8nPByZUAvi5upoW2qSKDQCVHX?usp=sharing)

## Locally
If you have a Linux machine, then you can just follow the Ctrl-G tutorial and adjust the parameters based on your resources.
However, if you are using Windows, then there are some extra steps that you need to follow.

One of the packages that Ctrl-G uses, [Triton](https://github.com/triton-lang/triton), is only supported on Linux.
This means that you have two options: use [WSL](https://learn.microsoft.com/en-us/windows/wsl/install) or remove the Triton dependency.

I'd reccommend trying out WSL, but if you don't want to deal with Linux, then here is how you can edit the code
so that it can run without Triton (thanks to Russell for figuring this out):
1. Clone the [Ctrl-G](https://github.com/joshuacnf/Ctrl-G/tree/main) repository
2. Follow the instructions and install the dependencies (I prefer pip, but you can also use conda as they suggested)
3. Open the ctrg/util.py file and change all the `@torch.compile` commands to `@torch.compile(backend="eager")`.
There should be 6 in total.
4. And that's it --- you can now run Ctrl-G on your local Windows machine!

