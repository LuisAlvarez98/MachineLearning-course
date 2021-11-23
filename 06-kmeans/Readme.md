# Installation

I've provided the requirements.txt file for easier **pip** installation.

```bash
   $ python3 -m venv env && source env/bin/activate
   $ pip install -r requirements.txt
```

Because I used a library for progress bar visualization, you'll also need to enable ipywidgets using the following command.

```bash
    $ jupyter nbextension enable --py widgetsnbextension
```