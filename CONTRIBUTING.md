# How to help ms2ml??

Right now the project is in development phase,
so I do not feel like it is ready to use BUT it is ready to recieve feedback.

Having said that,
I am happy to hear what improvements you would like from it so feel free to open an issue and let me know.

Issue types:

1. **Feature requests** / Wish list
   - Let me know of you use-case and what feature you feel should be included.
1. **Bug reports** to make one of these, please include:
   - a description of the problem
   - a way to reproduce the problem (files/file contents)
   - a description of the expected behavior
1. **General issues**
1. **Questions**

Most certainly PRs are welcome,
but I would encourage opening an issue first to discuss what would be implemented in the pull request.

## Setting up coding environment

This repo uses two main tools to make the developer experience nice.

1. Poetry
   - Poetry helps with the dependency management,
     installation,
     publishing, local environment ...

1. pre-commit
   - pre-commit helps run all the checks before they get to the main repository.
   - formats the code and files.
   - runs unit testing.
   - organizes dependencies
   - runs style checks ...

So if i wanted to start a development environment from scratch I would run ...

```shell
pip install pipx
pipx install poetry
pipx install pre-commit

# After forking the project
git clone https://github.com/{{YOURUSERNAME}}/ms2ml
cd ms2ml
pre-commit install
poetry shell
poetry install --all-extras
```

Then I could start editing ...
thoughout the edditing process I would run:

```
# To fix the styling
poetry run black ./THE_FILE_I_MODIFIED.py
poetry run isort ./THE_FILE_I_MODIFIED.py

# To look at some of the suggested coding conventions
poetry run ruff ./MODIFIED_FILE.py

# To run the unit tests
poetry run python -m pytest

```

Once you stage the files with `git add` you can run `pre-commit run` to run all the checks.

All checks will run again when you generate your commit with pre-commit and will ask you to add the files again before generating the commit.

## Coding standards

Most of the standards are enforced automatically but ...

1. `black` for style.
1. `isort` for import sorting.
1. `ruff` for basic linting.
1. `pylint` for additional linting.
1. `google` standard for docstrings.
1. `mdformat` for markdown document formatting.

## Documentation

There are three main ways of documenting the code,
all of them are related.

1. Formal documentation website.
   - This is built using mkdocs,
     so the markdown files where the long form documentation is kept as markdown files that generate the website are located in /docs

1. Docstrings
   - These are the documentation string that are written inside the code after a function/module/class/method definition.
   - These are also exported to the website documentation using mkdocstrings

1. Type hints
   - These are the type specifications that are given to arguments and function outputs,
     these are great at providing information to your code editor or IDE when using the library.
