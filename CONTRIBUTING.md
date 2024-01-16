# Contributing 

## Installing development versions

If you want to explore a potential code change, investigate
a bug, or just want to try unreleased features, you can also install
specific git shas.

Some example commands:

```shell
pip3 install git+https://git@github.com/pinecone-io/pinecone-python-client.git
pip3 install git+https://git@github.com/pinecone-io/pinecone-python-client.git@example-branch-name
pip3 install git+https://git@github.com/pinecone-io/pinecone-python-client.git@44fc7ed

poetry add git+https://github.com/pinecone-io/pinecone-python-client.git@44fc7ed
```


## Developing locally with Poetry 

[Poetry](https://python-poetry.org/) is a tool that combines [virtualenv](https://virtualenv.pypa.io/en/latest/) usage with dependency management, to provide a consistent experience for project maintainers and contributors who need to develop the pinecone-python-client
as a library. 

A common need when making changes to the Pinecone client is to test your changes against existing Python code or Jupyter Notebooks that `pip install` the Pinecone Python client as a library. 

Developers want to be able to see their changes to the library immediately reflected in their main application code, as well as to track all changes they make in git, so that they can be contributed back in the form of a pull request. 

The Pinecone Python client therefore supports Poetry as its primary means of enabling a consistent local development experience. This guide will walk you through the setup process so that you can: 
1. Make local changes to the Pinecone Python client that are separated from your system's Python installation
2. Make local changes to the Pinecone Python client that are immediately reflected in other local code that imports the pinecone client
3. Track all your local changes to the Pinecone Python client so that you can contribute your fixes and feature additions back via GitHub pull requests

### Step 1. Fork the Pinecone python client repository

On the [GitHub repository page](https://github.com/pinecone-io/pinecone-python-client) page, click the fork button at the top of the screen and create a personal fork of the repository: 

![Create a GitHub fork of the Pinecone Python client](./docs/pinecone-python-client-fork.png)

It will take a few seconds for your fork to be ready. When it's ready, **clone your fork** of the Pinecone python client repository to your machine. 

Change directory into the repository, as we'll be setting up a virtualenv from within the root of the repository. 

### Step 1. Install Poetry 

Visit [the Poetry site](https://python-poetry.org/) for installation instructions. 

### Step 2. Install dependencies 

Run `poetry install` from the root of the project. 

### Step 3. Activate the Poetry virtual environment and verify success

Run `poetry shell` from the root of the project. At this point, you now have a virtualenv set up in this directory, which you can verify by running: 

`poetry env info`

You should see something similar to the following output: 

```bash
Virtualenv
Python:         3.9.16
Implementation: CPython
Path:           /home/youruser/.cache/pypoetry/virtualenvs/pinecone-fWu70vbC-py3.9
Executable:     /home/youruser/.cache/pypoetry/virtualenvs/pinecone-fWu70vbC-py3.9/bin/python
Valid:          True

System
Platform:   linux
OS:         posix
Python:     3.9.16
Path:       /home/linuxbrew/.linuxbrew/opt/python@3.9
```
If you want to extract only the path to your new virtualenv, you can run `poetry env info --path`

## Loading your virtualenv in another shell 

It's a common need when developing against this client to load it as part of some other application or Jupyter Notebook code, modify 
it directly, see your changes reflected immediately and also have your changes tracked in git so you can contribute them back. 

It's important to understand that, by default, if you open a new shell or terminal window, or, for example, a new pane in a tmux session, 
your new shell will not yet reference the new virtualenv you created in the previous step. 

### Step 1. Get the path to your virtualenv

We're going to first get the path to the virtualenv we just created, by running: 

```bash
poetry env info --path
```

You'll get a path similar to this one:  `/home/youruser/.cache/pypoetry/virtualenvs/pinecone-fWu70vbC-py3.9/`

### Step 2. Load your existing virtualenv in your new shell

Within this path is a shell script that lives at `<your-virtualenv-path>/bin/activate`. Importantly, you cannot simply run this script, but you 
must instead source it like so: 

```bash
source /home/youruser/.cache/pypoetry/virtualenvs/pinecone-fWu70vbC-py3.9/bin/activate
```
In the above example, ensure you're using your own virtualenv path as returned by `poetry env info --path`.

### Step 3. Test out your virtualenv 

Now, we can test that our virtualenv is working properly by adding a new test module and function to the `pinecone` client within our virtualenv 
and running it from the second shell. 

#### Create a new test file in pinecone-python-client
In the root of your working directory of the `pinecone-python-client` where you first ran `poetry shell`, add a new file named `hello_virtualenv.py` under the `pinecone` folder. 

In that file write the following: 

```python
def hello():
    print("Hello, from your virtualenv!")
```
Save the file. 

#### Create a new test file in your second shell 
This step demonstrates how you can immediately test your latest Pinecone client code from any local Python application or Jupyter Notebook: 

In your second shell, where you ran `source` to load your virtualenv, create a python file named `test.py` and write the following: 

```python
from pinecone import hello_virtualenv

hello_virtualenv.hello()
```

Save the file. Run it with your Python binary. Depending on your system, this may either be `python` or `python3`: 

```bash
python3 test.py
```

You should see the following output: 

```bash
❯ python3 test.py
Hello, from your virtualenv!
```

If you experience any issues please [file a new issue](https://github.com/pinecone-io/pinecone-python-client/issues/new).
