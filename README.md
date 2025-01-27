# synapse
My toy deep learning framework to learn more in-depth how they work

## How to install

1. Install the requirements in the `requirements.txt` file
2. Optional: Run `cmake -S . -B build` to create a `compiler_commands.json` for your LSP
3. Run the command `pip install .` to install the development wheel
4. During development, it can be useful to instead install it with `pip install --no-build-isolation -Ceditable.rebuild=true -ve .` to automatically rebuild any code (if needed) whenever the package is imported into a python session

