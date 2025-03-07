import os
import glob
import importlib

def import_modules(models_dir, namespace, excludes=[]):
    for path in glob.glob(models_dir + "/**", recursive=True)[1:]:
        if any(e in path for e in excludes):
            continue

        file = os.path.split(path)[1]
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            module_name = file[: file.find(".py")] if file.endswith(".py") else file

            _namespace = path.replace("/", ".")
            _namespace = _namespace[
                _namespace.find(namespace) : _namespace.rfind("." + module_name)
            ]
            importlib.import_module(_namespace + "." + module_name)