# Contribution Guidelines

## Getting Started

To get started with contributions to the library follow these steps:

1. Fork the [repository](https://github.com/inferno-ml/inferno).
2. Clone your fork to your machine:
    ```sh
    git clone git@github.com:my-username/inferno.git
    ```
3. Install the Python package in an editable fashion with all development dependencies (in a new virtual environment):
    ```sh
    cd inferno
    pip install -e .[dev]
    ```

That's it! To make contributions to the main library simply push your changes to your fork on GitHub and create a pull request.

## Continuous Integration

We use [``tox``](https://tox.wiki) to simplify any tasks related to continuous integration.

### Running Tests
You can run (a subset of) tests via
```sh
tox -e py3 -- path-to-tests
```

### Formatting Code
To ensure consistent formatting throughout the library, we check for properly formatted code and imports. 

To check whether your code is properly formatted run:

```sh
tox -e format -- . --check --diff
```

If you want ``tox`` to automatically format a specific folder, simply run
```sh
tox -e format -- folder-to-format
```

### Building the Documentation
You can build the documentation and view it locally in your browser.

```sh
tox -e docs -- serve 
```

Now open a browser and enter the local address shown in your terminal.


### Running Code Examples
You can run the code examples in the documentation, like so.

```sh
# Run all examples
tox -e examples

# Run a specific example
tox -e examples -- docs/examples/my_example/run.py 
```