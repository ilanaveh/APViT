"""
IN 28/12/25
Added file for helper functions (e.g. override functions from mmcls that have problems related to versions etc.
"""
from yapf.yapflib.yapf_api import FormatCode


def pretty_text(cfg_dict):
    """
    Based on pretty_text property from Config Class in mmcv.utils.config.Config.pretty_text.
    The problem was that they used 'FormatCode(text, style_config=yapf_style, verify=True)', but 'verify' is no longer
    an argument for FormatCode.
    So this function is for independent usage (not within a Config instance)
        => instead of using self._cfg_dict, it gets cfg_dict directly.

    Changes:
        1. Arg cfg_dict instead of self.
        2. In accordance with 1, use cfg_dict instead of self._cfg_dict.
        3. Remove 'verify' from FormatCode arguments.

    Usage:
        pretty_text(cfg._cfg_dict.to_dict())
    """
    indent = 4

    def _indent(s_, num_spaces):
        s = s_.split('\n')
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(num_spaces * ' ') + line for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s

    def _format_basic_types(k, v, use_mapping=False):
        if isinstance(v, str):
            v_str = f"'{v}'"
        else:
            v_str = str(v)

        if use_mapping:
            k_str = f"'{k}'" if isinstance(k, str) else str(k)
            attr_str = f'{k_str}: {v_str}'
        else:
            attr_str = f'{str(k)}={v_str}'
        attr_str = _indent(attr_str, indent)

        return attr_str

    def _format_list(k, v, use_mapping=False):
        # check if all items in the list are dict
        if all(isinstance(_, dict) for _ in v):
            v_str = '[\n'
            v_str += '\n'.join(
                f'dict({_indent(_format_dict(v_), indent)}),'
                for v_ in v).rstrip(',')
            if use_mapping:
                k_str = f"'{k}'" if isinstance(k, str) else str(k)
                attr_str = f'{k_str}: {v_str}'
            else:
                attr_str = f'{str(k)}={v_str}'
            attr_str = _indent(attr_str, indent) + ']'
        else:
            attr_str = _format_basic_types(k, v, use_mapping)
        return attr_str

    def _contain_invalid_identifier(dict_str):
        contain_invalid_identifier = False
        for key_name in dict_str:
            contain_invalid_identifier |= \
                (not str(key_name).isidentifier())
        return contain_invalid_identifier

    def _format_dict(input_dict, outest_level=False):
        r = ''
        s = []

        use_mapping = _contain_invalid_identifier(input_dict)
        if use_mapping:
            r += '{'
        for idx, (k, v) in enumerate(input_dict.items()):
            is_last = idx >= len(input_dict) - 1
            end = '' if outest_level or is_last else ','
            if isinstance(v, dict):
                v_str = '\n' + _format_dict(v)
                if use_mapping:
                    k_str = f"'{k}'" if isinstance(k, str) else str(k)
                    attr_str = f'{k_str}: dict({v_str}'
                else:
                    attr_str = f'{str(k)}=dict({v_str}'
                attr_str = _indent(attr_str, indent) + ')' + end
            elif isinstance(v, list):
                attr_str = _format_list(k, v, use_mapping) + end
            else:
                attr_str = _format_basic_types(k, v, use_mapping) + end

            s.append(attr_str)
        r += '\n'.join(s)
        if use_mapping:
            r += '}'
        return r

    text = _format_dict(cfg_dict, outest_level=True)
    # copied from setup.cfg
    yapf_style = dict(
        based_on_style='pep8',
        blank_line_before_nested_class_or_def=True,
        split_before_expression_after_opening_paren=True)
    text, _ = FormatCode(text, style_config=yapf_style)
