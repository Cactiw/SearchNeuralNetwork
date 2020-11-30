

import xml.etree.ElementTree as ET

import pandas as pd

import logging
import traceback


def parse_XML(xml_str, columns, row_name, path: str = None):
    """get xml.etree root, the columns and return Pandas DataFrame"""
    if path is not None:
        root = ET.parse(path)
    else:
        root = ET.fromstring(xml_str)

    df = None
    try:
        xml_data = []
        rows = root.findall('.//{}'.format(row_name))
        for row in rows:
            pd_row = []
            for c in columns:
                try:
                    if c.startswith("<"):
                        # Специфические теги
                        c = c.replace("<", "").replace(">", "")
                        tag, name = c.split("=")
                        elem = row.find(tag).text
                    else:
                        elem = row.find(c)
                        if elem is not None:
                            elem = elem.text
                        else:
                            elem = row.get(c)
                except Exception:
                    logging.error("Error trying find elem {}: {}".format(c, traceback.format_exc()))
                    elem = None
                pd_row.append(elem)
            xml_data.append(pd_row)

        df = pd.DataFrame(xml_data, columns=columns)
    except Exception as e:
        logging.error('[xml_to_pandas] Exception: {}.'.format(e))

    return df
