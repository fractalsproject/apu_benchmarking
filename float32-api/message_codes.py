CODE_INDEX = 1
STR_INDEX = 0
GENERAL_ERROR = ("server error", "GENERAL_ERROR")
OK = ("OK", "")


class Messages:
    def __init__(self):
        pass

    @staticmethod
    def get_str_from_err(err_ojb):
        return err_ojb[STR_INDEX]

    @staticmethod
    def get_dict(err_ojb):
        return {err_ojb[CODE_INDEX]: err_ojb[STR_INDEX]}


