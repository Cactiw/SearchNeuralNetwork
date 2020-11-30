
import re

trim_all_pattern = re.compile("^\n\\s+(.+)\\n\\s*$", re.DOTALL)


def trim_all(s: str) -> str:
    parse = re.match(trim_all_pattern, s)
    if parse is None:
        return s
    return parse.group(1)

