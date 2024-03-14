import copy
from dataclasses import field


def warn_default(key, val):
    if val != "None":
        print(
            f"""! key NOT FOUND inside parameters.json: "{key}"\n\t\t  Default value used: {val}"""
        )
    return val


def warn_pop(dico: dict, key: str, default):
    if dico.get(key):
        return dico.pop(key, default)
    return warn_default(key, default)


def set_default(key: str, val):
    # pylint: disable=invalid-field-call
    return field(default_factory=lambda: warn_default(key, val))


def deep_dict_update(main_dict: dict, new_dict: dict):
    """Update recursively a nested dict with another.
    main_dict keys/values that do not exist in new_dict are kept.

    Parameters
    ----------
    main_dict : dict
        Main dict with all default values
    new_dict : dict
        Dict with new values to update

    Returns
    -------
    dict
        The main_dict overwrite by new_dict value
    """
    main_deep_copy = copy.deepcopy(main_dict)
    for key, value in new_dict.items():
        if isinstance(value, dict):
            main_deep_copy[key] = deep_dict_update(main_deep_copy.get(key, {}), value)
        else:
            main_deep_copy[key] = value
    return main_deep_copy


def merge_common_and_labels(raw_params, label):
    common_params = raw_params["common"]
    if label.value not in raw_params["labels"]:
        return common_params
    label_params = raw_params["labels"][label.value]
    return deep_dict_update(common_params, label_params)
