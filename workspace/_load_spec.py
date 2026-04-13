import accelforge as af
import os

THIS_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARCH_DIR = os.path.join(THIS_SCRIPT_DIR, "arch")


def get_spec(
    arch_name: str,
    add_dummy_main_memory: bool = False,
    jinja_parse_data: dict = None,
) -> af.Spec:
    """Load an architecture spec from YAML.

    Parameters
    ----------
    arch_name : str
        Architecture name (without .yaml extension).
    add_dummy_main_memory : bool
        Whether to add a dummy main memory for DNN workloads.
    jinja_parse_data : dict
        Jinja template variables for the YAML file.

    Returns
    -------
    af.Spec
    """
    arch_path = os.path.join(ARCH_DIR, f"{arch_name}.yaml")
    jinja_data = jinja_parse_data or {}

    variables = af.Variables.from_yaml(
        arch_path, top_key="variables", jinja_parse_data=jinja_data
    )
    arch = af.Arch.from_yaml(
        arch_path, top_key="arch", jinja_parse_data=jinja_data
    )
    # Workload is optional — arch YAMLs that don't ship their own workload
    # (e.g. basic8.yaml) expect the caller to inject one after get_spec returns.
    workload = None
    try:
        workload = af.Workload.from_yaml(
            arch_path, top_key="workload", jinja_parse_data=jinja_data
        )
    except KeyError:
        pass

    renames_data = {}
    try:
        renames_data = af.Renames.from_yaml(
            arch_path, top_key="renames", jinja_parse_data=jinja_data
        )
    except Exception:
        pass

    spec_kwargs = dict(arch=arch, variables=variables, renames=renames_data)
    if workload is not None:
        spec_kwargs["workload"] = workload
    spec = af.Spec(**spec_kwargs)

    spec.config.expression_custom_functions.append(
        os.path.join(ARCH_DIR, "_include_functions.py")
    )
    spec.config.component_models.append(
        os.path.join(ARCH_DIR, "components/*.py")
    )

    spec.mapper._let_non_intermediate_tensors_respawn_in_backing_storage = True

    if add_dummy_main_memory:
        main_memory = af.arch.Memory(
            name="MainMemory",
            component_class="Dummy",
            size=float("inf"),
            tensors={"keep": "~weight"},
        )
        spec.arch.nodes.insert(0, main_memory)

    return spec
